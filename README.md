# auria-execution

Expert assembly and execution pipeline for AURIA Runtime Core.

## Overview

Implements the core execution engine that orchestrates expert assembly and forward pass execution.

## Execution Backend Trait

```rust
use auria_core::{ExecutionOutput, ExecutionState, Result, Tensor};
use async_trait::async_trait;

#[async_trait]
pub trait ExecutionBackend: Send + Sync {
    async fn execute_step(
        &self,
        input: Tensor,
        experts: Vec<Tensor>,
        state: ExecutionState,
    ) -> Result<ExecutionOutput>;

    fn backend_name(&self) -> &str;
}
```

## Execution Flow

1. Router selects experts
2. Expert Cache lookup
3. Expert Assembler assembles missing experts
4. Execution Core executes forward pass
5. Tokens returned to user

## Usage

```rust
use auria_execution::ExecutionEngine;

let engine = ExecutionEngine::new(backend);
let output = engine.execute(input, routing, state).await?;
```
