// File: lib.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     Expert assembly and execution pipeline for AURIA Runtime Core.
//     Implements the core execution engine that orchestrates expert assembly
//     and forward pass execution across different hardware backends.
//
use auria_core::{ExecutionOutput, ExecutionState, RoutingDecision, Tensor, AuriaResult};
use async_trait::async_trait;

#[async_trait]
pub trait ExecutionBackend: Send + Sync {
    async fn execute_step(
        &self,
        input: Tensor,
        experts: Vec<Tensor>,
        state: ExecutionState,
    ) -> AuriaResult<ExecutionOutput>;

    fn backend_name(&self) -> &str;
}

pub struct ExecutionEngine<B: ExecutionBackend> {
    backend: B,
}

impl<B: ExecutionBackend> ExecutionEngine<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub async fn execute(
        &self,
        input: Tensor,
        routing: RoutingDecision,
        state: ExecutionState,
    ) -> AuriaResult<ExecutionOutput> {
        let experts: Vec<Tensor> = Vec::new();
        self.backend.execute_step(input, experts, state).await
    }
}
