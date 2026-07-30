import httpx

from fastapi import FastAPI, HTTPException, Request
from openai import OpenAI

from copilot.schemas.query import QueryRequest, QueryResponse
from copilot.api.settings import ApiSettings, load_api_settings
from copilot.api.query_service import query_service
from copilot.storage.database import (
    create_engine_from_url,
    create_session_factory,
)
from copilot.api.errors import (
    ManifestNotConfiguredError,
    ManifestFileNotFoundError,
    InvalidManifestError,
    DatabaseNotConfiguredError,
)
from copilot.schemas.triage import (
    TriageRequest,
    TriageReport,
)
from copilot.services.triage import (
    TriageService,
    TriageRunNotFoundError,
    TriageServiceError,
)
from copilot.tools.anomaly import AnomalyOperationalTools
from copilot.clients.anomaly_api import AnomalyApiClient
from copilot.providers.factory import (
    create_embedding_provider,
    create_grounded_answer_generator,
)
from copilot.providers.errors import (
    GroundedAnswerProviderError,
    InvalidGroundedAnswerResponseError,
)


def create_app(
    settings: ApiSettings | None = None,
    anomaly_transport: httpx.BaseTransport | None = None,
    openai_client: OpenAI | None = None,
) -> FastAPI:
    app = FastAPI(
        title="AnomalyOps-Copilot API",
        description="API RAG answers.",
        version="0.1.0",
    )
    
    resolved_settings = settings or ApiSettings()
    
    app.state.settings = resolved_settings
    app.state.database_engine = None
    app.state.session_factory = None
    app.state.anomaly_client = None
    app.state.anomaly_tools = None
    app.state.triage_service = None
    app.state.embedding_provider = None
    app.state.grounded_answer_generator = None
    
    if (
        resolved_settings.retrieval_backend == "pgvector"
        and resolved_settings.database_url is not None
    ):
        engine = create_engine_from_url(resolved_settings.database_url)
        
        app.state.database_engine = engine
        app.state.session_factory = create_session_factory(engine)
        
        app.state.embedding_provider = create_embedding_provider(
            resolved_settings,
            openai_client=openai_client,
        )
        
    if resolved_settings.anomaly_api_base_url is not None:
        anomaly_client = AnomalyApiClient(
            resolved_settings.anomaly_api_base_url,
            transport=anomaly_transport,
        )
        anomaly_tools = AnomalyOperationalTools(anomaly_client)
        
        app.state.anomaly_client = anomaly_client
        app.state.anomaly_tools = anomaly_tools
        app.state.triage_service = TriageService(
            app.state.anomaly_tools
        )
        
    app.state.grounded_answer_generator = create_grounded_answer_generator(
        resolved_settings,
        openai_client=openai_client,
    )
        

    @app.get("/health")
    def health_check():
        return {"status": "ok"}


    @app.post("/query", response_model=QueryResponse)
    def query(request: Request, query_request: QueryRequest) -> QueryResponse:
        settings = request.app.state.settings
        embedding_provider = request.app.state.embedding_provider
        
        try:
            query_response = query_service(
                settings,
                query_request,
                session_factory=request.app.state.session_factory,
                embedding_provider=embedding_provider,
                grounded_answer_generator=request.app.state.grounded_answer_generator,
            )
        except ManifestNotConfiguredError:
            raise HTTPException(
                status_code=500,
                detail="Manifest path is not configured.",
            )
        except ManifestFileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="Manifest file was not found."
            )
        except InvalidManifestError:
            raise HTTPException(
                status_code=500,
                detail="Manifest file is invalid."
            )
        except DatabaseNotConfiguredError:
            raise HTTPException(
                status_code=500,
                detail="Database URL is not configured."
            )
        except InvalidGroundedAnswerResponseError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Grounded answer provider returned an invalid response."
                ),
            ) from exc
        except GroundedAnswerProviderError as exc:
            raise HTTPException(
                status_code=502,
                detail="Grounded answer provider request failed.",
            ) from exc
            
        return query_response
    
    
    @app.post("/triage", response_model=TriageReport)
    def post_triage_report(request: Request, triage_request: TriageRequest) -> TriageReport:
        service = request.app.state.triage_service
        
        if service is None:
            raise HTTPException(
                status_code=500,
                detail="Anomaly API base URL is not configured."
            )
            
        try:
            report = service.triage(request=triage_request)
        except TriageRunNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="The requested run was not found.",
            ) from exc
        except TriageServiceError as exc:
            raise HTTPException(
                status_code=502,
                detail="Anomaly API request failed.",
            ) from exc
            
        return report
    
    
    @app.on_event("shutdown")
    def shutdown_database_engine() -> None:
        engine = app.state.database_engine
        client = app.state.anomaly_client
        
        if client is not None:
            client.close()
        
        if engine is not None:
            engine.dispose()
        

    return app


app = create_app(settings=load_api_settings())