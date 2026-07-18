from fastapi import FastAPI, HTTPException, Request

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


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    app = FastAPI(
        title="AnomalyOps-Copilot API",
        description="API RAG answers.",
        version="0.1.0",
    )
    
    resolved_settings = settings or ApiSettings()
    
    app.state.settings = resolved_settings
    app.state.database_engine = None
    app.state.session_factory = None
    
    if (
        resolved_settings.retrieval_backend == "pgvector"
        and settings.database_url is not None
    ):
        engine = create_engine_from_url(resolved_settings.database_url)
        
        app.state.database_engine = engine
        app.state.session_factory = create_session_factory(engine)
        

    @app.get("/health")
    def health_check():
        return {"status": "ok"}


    @app.post("/query", response_model=QueryResponse)
    def query(request: Request, query_request: QueryRequest) -> QueryResponse:
        settings = request.app.state.settings
        
        try:
            query_response = query_service(
                settings,
                query_request,
                session_factory=request.app.state.session_factory,
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
            
        return query_response
    
    
    @app.on_event("shutdown")
    def shutdown_database_engine() -> None:
        engine = app.state.database_engine
        
        if engine is not None:
            engine.dispose()
        

    return app


app = create_app(settings=load_api_settings())