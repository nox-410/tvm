/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
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
 *  Optional module when build cuda is switched to off
 */
#include "../../runtime/cuda/cuda_module.h"

#include "../build_common.h"
#include "../source/codegen_cuda.h"

namespace tvm {
namespace runtime {

class CUDAModuleNode : public runtime::ModuleNode {
 public:
  explicit CUDAModuleNode(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string cuda_source)
      : data_(data), fmt_(fmt), fmap_(fmap), cuda_source_(cuda_source) {
  }

  ~CUDAModuleNode() {}

  const char* type_key() const final { return "cuda"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
      ICHECK(0) << "Not implemented when cuda is not enabled in TVM.";
      return PackedFunc();
  }

  std::string GetSource(const std::string& format) final {
    if (format == fmt_) return data_;
    if (cuda_source_.length() != 0) {
      return cuda_source_;
    } else {
      if (fmt_ == "ptx") return data_;
      return "";
    }
  }

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The cuda source.
  std::string cuda_source_;
};

Module CUDAModuleCreate(std::string data, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap,
                        std::string cuda_source) {
  auto n = make_object<CUDAModuleNode>(data, fmt, fmap, cuda_source);
  return Module(n);
}

} // namespace runtime
} // namespace tvm

namespace tvm {
namespace codegen {

runtime::Module BuildCUDA(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenCUDA cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenCUDA: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenCUDA: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_cuda_postproc")) {
    code = (*f)(code).operator std::string();
  }
  const auto* f_enter = Registry::Get("target.TargetEnterScope");
  (*f_enter)(target);
  const auto* f_exit = Registry::Get("target.TargetExitScope");
  (*f_exit)(target);
  return CUDAModuleCreate("", "ptx", ExtractFuncInfo(mod), code);
}

TVM_REGISTER_GLOBAL("target.build.cuda").set_body_typed(BuildCUDA);

}  // namespace codegen
}  // namespace tvm
