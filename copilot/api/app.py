from fastapi import FastAPI, HTTPException, Request

from copilot.schemas.query import QueryRequest, QueryResponse
from copilot.api.settings import ApiSettings, load_api_settings
from copilot.api.query_service import query_service


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    app = FastAPI(
        title="AnomalyOps-Copilot API",
        description="API RAG answers.",
        version="0.1.0",
    )
    
    app.state.settings = settings or ApiSettings()
        

    @app.get("/health")
    def health_check():
        return {"status": "ok"}


    @app.post("/query", response_model=QueryResponse)
    def query(request: Request, query_request: QueryRequest) -> QueryResponse:
        settings = request.app.state.settings
        manifest_path = settings.manifest_path
        
        try:
            query_response = query_service(manifest_path, query_request)
        except ValueError:
            raise HTTPException(
                status_code=500,
                detail="Manifest path is not configured.",
            )
            
        return query_response
        

    return app


app = create_app(settings=load_api_settings())