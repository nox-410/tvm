/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tile costmodel.cc
 * \brief Estimate the program IR with roofline cost model.
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../op/op.h"

namespace tvm {
namespace tl {

using namespace tir;

struct OpCostGlobal {
  OpCost cost;
  int64_t num_block;
};

class OpCostVisitor : public arith::IRMutatorWithAnalyzer {
 public:
  static OpCostGlobal Collect(const PrimFunc& f) {
    arith::Analyzer analyzer;
    OpCostVisitor T(&analyzer);
    T.target_ = f->GetAttr<Target>(tvm::attr::kTarget).value();
    for (const auto& [_, buffer] : f->buffer_map) {
      T.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    T(f->body);
    OpCostGlobal cost_global;
    cost_global.cost = T.op_cost_sum_;
    cost_global.num_block = T.grid_extent_;
    return cost_global;
  }

 private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const ForNode* node) final {
    if (node->kind == ForKind::kParallel) {
      return GetRef<Stmt>(node);
    } else {
      auto loop_bound = analyzer_->const_int_bound(node->extent);
      ICHECK(loop_bound->max_value != arith::ConstIntBound::kPosInf ||
             loop_bound->min_value != arith::ConstIntBound::kNegInf)
          << loop_bound;
      int64_t loop_mean_value = (loop_bound->max_value + loop_bound->min_value) / 2;
      cur_loop_extent_ *= loop_mean_value;
      auto n = arith::IRMutatorWithAnalyzer::VisitStmt_(node);
      cur_loop_extent_ /= loop_mean_value;
      return n;
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* node) final {
    if (node->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(node->node);
      if (iv->thread_tag == "blockIdx.x" || iv->thread_tag == "blockIdx.y" ||
          iv->thread_tag == "blockIdx.z") {
        auto extent_ptr = as_const_int(iv->dom->extent);
        ICHECK(extent_ptr);
        grid_extent_ *= *extent_ptr;
      } else if (iv->thread_tag == "threadIdx.x") {
        auto extent_ptr = as_const_int(iv->dom->extent);
        ICHECK(extent_ptr);
        block_extent_ = *extent_ptr;
      }
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(node);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    for (auto buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode* node) final {
    auto stmt = arith::IRMutatorWithAnalyzer::VisitStmt_(node);
    auto op = ParseOperator(stmt, buffer_data_to_buffer_);
    if (op == nullptr) return stmt;
    OpCost cost = op->GetOpCost(target_, block_extent_, analyzer_);
    op_cost_sum_ = op_cost_sum_ + cost * cur_loop_extent_;
    return stmt;
  }

  int64_t cur_loop_extent_ = 1;
  int64_t grid_extent_ = 1;
  int64_t block_extent_ = 1;
  BufferMap buffer_data_to_buffer_;
  Target target_;
  OpCost op_cost_sum_;
};

extern Array<Buffer> DetectPipelinedBuffers(PrimFunc f);

class MemCap {
 public:
  MemCap(int64_t smem, int64_t regs, int64_t num_threads)
      : smem_(smem), regs_(regs), num_threads_(num_threads) {}
  int64_t smem_, regs_, num_threads_;
};

class MemUseCollector final : public StmtExprVisitor {
 public:
  static MemCap Collect(const PrimFunc& f) {
    MemUseCollector collector;
    auto pipelined_buffers = DetectPipelinedBuffers(f);
    for (auto buffer : pipelined_buffers) {
      collector.pipelined_buffers_.insert(buffer);
    }
    collector(f->body);
    return collector.ComputeMemCap();
  }

 private:
  /*! \brief record the touch list of statement. */
  struct StmtEntry {
    // The statement
    const Object* stmt;
    // The index in the linear_seq_ to point to end of the nested scope.
    // This is only set to non-zero if stmt is a nested scope.
    // if offset > 0, means this is the begin, the end entry is current_index + offset
    // if offset < 0, means this is the end, the begin entry is current_index + offset
    int64_t scope_pair_offset{0};
    // The buffer variables this statement touched.
    std::vector<const VarNode*> touched;
  };

  struct EventEntry {
    // variables we generate
    std::vector<const VarNode*> gen;
    // variables we kill
    std::vector<const VarNode*> kill;
  };

  void VisitStmt_(const BlockNode* node) final {
    for (auto buf : node->alloc_buffers) {
      buffer_data_to_buffer_.Set(buf->data, buf);
    }
    VisitStmt(node->body);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    StmtExprVisitor::VisitStmt_(op);
    // Add write access.
    const VarNode* buf = op->buffer->data.get();
    auto it = buffer_data_to_buffer_.find(GetRef<Var>(buf));
    if (it != buffer_data_to_buffer_.end()) {
      scope_.back().touched.push_back(buf);
    }
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (e.touched.size() != 0) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
  }

  void VisitStmt_(const EvaluateNode* op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    StmtExprVisitor::VisitStmt_(op);
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (e.touched.size() != 0) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    // Add write access.
    StmtExprVisitor::VisitExpr_(op);
    const VarNode* buf = op->buffer->data.get();
    auto it = buffer_data_to_buffer_.find(GetRef<Var>(buf));
    if (it != buffer_data_to_buffer_.end()) {
      scope_.back().touched.push_back(buf);
    }
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      const VarNode* buf = op->args[1].as<VarNode>();
      auto it = buffer_data_to_buffer_.find(GetRef<Var>(buf));
      if (it != buffer_data_to_buffer_.end()) {
        scope_.back().touched.push_back(buf);
      }
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  void VisitExpr_(const VarNode* var) final {
    // Directly reference to the variable count as a read.
    auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
    if (it != buffer_data_to_buffer_.end()) {
      scope_.back().touched.push_back(var);
    }
  }

  template <typename T>
  void VisitNewScope(const T* op) {
    scope_.push_back(StmtEntry());
    StmtEntry e;
    e.stmt = op;
    int64_t begin_index = static_cast<int64_t>(linear_seq_.size());
    // before scope.
    linear_seq_.push_back(e);
    StmtExprVisitor::VisitStmt_(op);
    // after scope.
    e.touched = std::move(scope_.back().touched);
    scope_.pop_back();
    int64_t end_index = static_cast<int64_t>(linear_seq_.size());
    ICHECK_GT(end_index, begin_index);
    e.scope_pair_offset = begin_index - end_index;
    linear_seq_.push_back(e);
    // record the pointer to end index.
    ICHECK_NE(end_index, 0U);
    linear_seq_[begin_index].scope_pair_offset = end_index - begin_index;
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        auto num_threads_p = as_const_int(iv->dom->extent);
        ICHECK(num_threads_p != nullptr) << "Extect static num threads here";
        num_threads = *num_threads_p;
      }
    }
    VisitNewScope(op);
  }
  void VisitStmt_(const IfThenElseNode* op) final { VisitNewScope(op); }
  void VisitStmt_(const ForNode* op) final {
    auto num_stages_anno = op->annotations.Get("num_stages");
    if (num_stages_anno.defined()) num_stages_ = num_stages_anno.as<IntImmNode>()->value;
    VisitNewScope(op);
  }
  void VisitStmt_(const WhileNode* op) final { VisitNewScope(op); }
  void VisitStmt_(const AssertStmtNode* op) final { VisitNewScope(op); }

  std::unordered_map<const Object*, EventEntry> MakeEventMap() {
    // find kill point, do a reverse linear scan.
    std::unordered_set<const VarNode*> touched;
    std::unordered_map<const Object*, EventEntry> event_map;
    for (size_t i = linear_seq_.size(); i != 0; --i) {
      const StmtEntry& s = linear_seq_[i - 1];
      for (const VarNode* buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          event_map[s.stmt].kill.push_back(buffer);
        }
      }
    }
    // find gen point, do forward scan
    touched.clear();
    for (size_t i = 0; i < linear_seq_.size(); ++i) {
      int64_t offset = linear_seq_[i].scope_pair_offset;
      if (offset < 0) continue;
      const StmtEntry& s = linear_seq_[i + offset];
      for (const VarNode* buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          event_map[s.stmt].gen.push_back(buffer);
        }
      }
    }
    return event_map;
  }

  int64_t GetBufferbytes(Buffer buffer) {
    int bytes = buffer->dtype.bytes();
    PrimExpr num_elems = 1;
    for (auto dim : buffer->shape) num_elems *= dim;
    auto int_p = as_const_int(num_elems * bytes);
    ICHECK(int_p != nullptr) << "Expect static shape.";
    if (pipelined_buffers_.count(buffer))
      return *int_p * num_stages_;
    else
      return *int_p;
  }

  MemCap ComputeMemCap() {
    auto event_map = MakeEventMap();
    int64_t regs = 0;
    int64_t smem = 0;
    int64_t max_regs = 0;
    int64_t max_smem = 0;
    for (auto seq : linear_seq_) {
      for (auto var : event_map[seq.stmt].gen) {
        auto buffer = buffer_data_to_buffer_[GetRef<Var>(var)];
        auto bytes = GetBufferbytes(buffer);
        if (buffer.scope() == "shared" || buffer.scope() == "shared.dyn") {
          smem += bytes;
          max_smem = std::max(smem, max_smem);
        } else if (buffer.scope() == "local") {
          regs += bytes;
          max_regs = std::max(regs, max_regs);
        } else if (buffer.scope() == "local.fragment") {
          regs += bytes / num_threads;
          max_regs = std::max(regs, max_regs);
        } else {
          ICHECK(0);
        }
      }
      for (auto var : event_map[seq.stmt].kill) {
        auto buffer = buffer_data_to_buffer_[GetRef<Var>(var)];
        auto bytes = GetBufferbytes(buffer);
        if (buffer.scope() == "shared" || buffer.scope() == "shared.dyn") {
          smem = smem - bytes;
        } else if (buffer.scope() == "local") {
          regs = regs - bytes;
        } else if (buffer.scope() == "local.fragment") {
          regs = regs - bytes / num_threads;
        } else {
          ICHECK(0);
        }
      }
    }
    return MemCap(max_smem, max_regs, num_threads);
  }

  int64_t num_threads;
  Map<Var, Buffer> buffer_data_to_buffer_;
  // linearized access sequence.
  std::vector<StmtEntry> linear_seq_;
  // The scope stack.
  std::vector<StmtEntry> scope_;

  std::unordered_set<Buffer, ObjectHash, ObjectEqual> pipelined_buffers_;
  int64_t num_stages_;
};

TVM_REGISTER_GLOBAL("tl.highlevel_costmodel").set_body_typed([](PrimFunc func) {
  OpCostGlobal cost_global = OpCostVisitor::Collect(func);
  OpCost cost = cost_global.cost;

  double tc_tflops = 170.0 * 1e9;
  double tc_times_ms = cost_global.num_block * cost.get(OpCost::kFp16TensorCore) / tc_tflops;

  std::cout << "Num blocks: " << cost_global.num_block << std::endl;
  std::cout << "Start OpCost Fields." << std::endl;
  std::cout << "  Cost gmem access: " << cost.get(OpCost::kGmemAccess) << std::endl;
  std::cout << "  Cost smem access: " << cost.get(OpCost::kSmemAccess) << std::endl;
  std::cout << "  Cost tensorcore : " << cost.get(OpCost::kFp16TensorCore);
  std::cout << " roofline " << std::setprecision(2) << tc_times_ms << " ms" << std::endl;
  std::cout << "  Cost flaot inst: " << cost.get(OpCost::kFloatSIMT) << std::endl;
  std::cout << "End OpCost Fields." << std::endl;

  auto mem = MemUseCollector::Collect(func);

  int reg_occupy = (64 << 10) / (mem.num_threads_ * (mem.regs_ / 4));
  int smem_occupy = (96 << 10) / mem.smem_;
  int active_blocks = std::min(reg_occupy, smem_occupy);
  int active_warps = (mem.num_threads_ / 32) * active_blocks;

  std::cout << "Use regs: " << (mem.regs_ / 4) << std::endl;
  std::cout << "Use smem: " << mem.smem_ << std::endl;
  std::cout << "Active warps: " << active_warps << std::endl;
  std::cout << "Active blocks: " << active_blocks << std::endl;

  return 0;
});

}  // namespace tl
}  // namespace tvm