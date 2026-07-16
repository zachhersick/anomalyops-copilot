from unittest.mock import MagicMock, patch
from pathlib import Path

from scripts.store_chunk_manifest import main

def test_main_stored_chunk_manifest(capsys):
    chunks = [MagicMock(), MagicMock()]
    engine = MagicMock()
    session = MagicMock()
    session_factory = MagicMock()
    
    session_factory.return_value.__enter__.return_value = session
    
    with (
        patch(
            "scripts.store_chunk_manifest.load_dotenv",
        ),
        patch(
            "scripts.store_chunk_manifest.os.getenv",
            return_value="postgresql+psychopg://test",
        ),
        patch(
            "scripts.store_chunk_manifest.load_chunk_manifest",
            return_value=chunks,
        ) as load_chunk_manifest,
        patch(
            "scripts.store_chunk_manifest.create_engine_from_url",
            return_value=engine,
        ) as create_engine_from_url,
        patch(
            "scripts.store_chunk_manifest.initialize_database",
        ) as initialize_database,
        patch(
            "scripts.store_chunk_manifest.create_session_factory",
            return_value=session_factory,
        ) as create_session_factory,
        patch(
            "scripts.store_chunk_manifest.store_source_chunks",
            return_value=2,
        ) as store_source_chunks,
    ):
        result = main(["outputs/chunks.json"])
    
    assert result == 0
    
    load_chunk_manifest.assert_called_once_with(
        Path("outputs/chunks.json")
    )
    create_engine_from_url.assert_called_once_with(
        "postgresql+psychopg://test"
    )
    initialize_database.assert_called_once_with(engine)
    create_session_factory.assert_called_once_with(engine)
    store_source_chunks.assert_called_once_with(session, chunks)
    
    assert capsys.readouterr().out == "Stored 2 source chunks.\n"