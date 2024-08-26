// This file is @generated by prost-build.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadyRequest {}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadyResponse {
    #[prost(bool, tag = "1")]
    pub ready: bool,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProveCoreRequest {
    #[prost(bytes = "vec", tag = "1")]
    pub data: ::prost::alloc::vec::Vec<u8>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProveCoreResponse {
    #[prost(bytes = "vec", tag = "1")]
    pub result: ::prost::alloc::vec::Vec<u8>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CompressRequest {
    #[prost(bytes = "vec", tag = "1")]
    pub data: ::prost::alloc::vec::Vec<u8>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CompressResponse {
    #[prost(bytes = "vec", tag = "1")]
    pub result: ::prost::alloc::vec::Vec<u8>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShrinkRequest {
    #[prost(bytes = "vec", tag = "1")]
    pub data: ::prost::alloc::vec::Vec<u8>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShrinkResponse {
    #[prost(bytes = "vec", tag = "1")]
    pub result: ::prost::alloc::vec::Vec<u8>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WrapRequest {
    #[prost(bytes = "vec", tag = "1")]
    pub data: ::prost::alloc::vec::Vec<u8>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WrapResponse {
    #[prost(bytes = "vec", tag = "1")]
    pub result: ::prost::alloc::vec::Vec<u8>,
}
pub use twirp;
pub const SERVICE_FQN: &str = "/api.ProverService";
#[twirp::async_trait::async_trait]
pub trait ProverService {
    async fn ready(
        &self,
        ctx: twirp::Context,
        req: ReadyRequest,
    ) -> Result<ReadyResponse, twirp::TwirpErrorResponse>;
    async fn prove_core(
        &self,
        ctx: twirp::Context,
        req: ProveCoreRequest,
    ) -> Result<ProveCoreResponse, twirp::TwirpErrorResponse>;
    async fn compress(
        &self,
        ctx: twirp::Context,
        req: CompressRequest,
    ) -> Result<CompressResponse, twirp::TwirpErrorResponse>;
    async fn shrink(
        &self,
        ctx: twirp::Context,
        req: ShrinkRequest,
    ) -> Result<ShrinkResponse, twirp::TwirpErrorResponse>;
    async fn wrap(
        &self,
        ctx: twirp::Context,
        req: WrapRequest,
    ) -> Result<WrapResponse, twirp::TwirpErrorResponse>;
}
pub fn router<T>(api: std::sync::Arc<T>) -> twirp::Router
where
    T: ProverService + Send + Sync + 'static,
{
    twirp::details::TwirpRouterBuilder::new(api)
        .route(
            "/Ready",
            |api: std::sync::Arc<T>, ctx: twirp::Context, req: ReadyRequest| async move {
                api.ready(ctx, req).await
            },
        )
        .route(
            "/ProveCore",
            |api: std::sync::Arc<T>, ctx: twirp::Context, req: ProveCoreRequest| async move {
                api.prove_core(ctx, req).await
            },
        )
        .route(
            "/Compress",
            |api: std::sync::Arc<T>, ctx: twirp::Context, req: CompressRequest| async move {
                api.compress(ctx, req).await
            },
        )
        .route(
            "/Shrink",
            |api: std::sync::Arc<T>, ctx: twirp::Context, req: ShrinkRequest| async move {
                api.shrink(ctx, req).await
            },
        )
        .route(
            "/Wrap",
            |api: std::sync::Arc<T>, ctx: twirp::Context, req: WrapRequest| async move {
                api.wrap(ctx, req).await
            },
        )
        .build()
}
#[twirp::async_trait::async_trait]
pub trait ProverServiceClient: Send + Sync + std::fmt::Debug {
    async fn ready(
        &self,
        req: ReadyRequest,
    ) -> Result<ReadyResponse, twirp::ClientError>;
    async fn prove_core(
        &self,
        req: ProveCoreRequest,
    ) -> Result<ProveCoreResponse, twirp::ClientError>;
    async fn compress(
        &self,
        req: CompressRequest,
    ) -> Result<CompressResponse, twirp::ClientError>;
    async fn shrink(
        &self,
        req: ShrinkRequest,
    ) -> Result<ShrinkResponse, twirp::ClientError>;
    async fn wrap(&self, req: WrapRequest) -> Result<WrapResponse, twirp::ClientError>;
}
#[twirp::async_trait::async_trait]
impl ProverServiceClient for twirp::client::Client {
    async fn ready(
        &self,
        req: ReadyRequest,
    ) -> Result<ReadyResponse, twirp::ClientError> {
        let url = self.base_url.join("api.ProverService/Ready")?;
        self.request(url, req).await
    }
    async fn prove_core(
        &self,
        req: ProveCoreRequest,
    ) -> Result<ProveCoreResponse, twirp::ClientError> {
        let url = self.base_url.join("api.ProverService/ProveCore")?;
        self.request(url, req).await
    }
    async fn compress(
        &self,
        req: CompressRequest,
    ) -> Result<CompressResponse, twirp::ClientError> {
        let url = self.base_url.join("api.ProverService/Compress")?;
        self.request(url, req).await
    }
    async fn shrink(
        &self,
        req: ShrinkRequest,
    ) -> Result<ShrinkResponse, twirp::ClientError> {
        let url = self.base_url.join("api.ProverService/Shrink")?;
        self.request(url, req).await
    }
    async fn wrap(&self, req: WrapRequest) -> Result<WrapResponse, twirp::ClientError> {
        let url = self.base_url.join("api.ProverService/Wrap")?;
        self.request(url, req).await
    }
}