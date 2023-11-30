/* Copyright 2023 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "jaxlib/mlir/_mlir_libs/jax_dialects.h"

#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

extern "C" {

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SCF, scf, mlir::scf::SCFDialect)

void mlirRegisterMemRefPasses() {
  mlir::memref::registerMemRefPasses();
}

void jaxMlirRegisterInterfaceExternalModels(MlirDialectRegistry registry) {
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(*unwrap(registry));
  mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(*unwrap(registry));
  mlir::registerGPUDialectTranslation(*unwrap(registry));
  mlir::registerLLVMDialectTranslation(*unwrap(registry));
  mlir::registerNVVMDialectTranslation(*unwrap(registry));
}

}
