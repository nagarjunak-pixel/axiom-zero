"""End-to-end tests for FastAPI endpoints, CLI commands, and real/replay LLM CEGIS benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from axiom_zero.api.app import create_app
from axiom_zero.cli import app as cli_app
from axiom_zero.real_llm_eval import (
    build_hard_formal_suite,
    run_real_ollama_benchmark,
    verify_candidate_json,
)

runner = CliRunner()


def test_verify_candidate_json_and_offline_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = build_hard_formal_suite()
    t0 = suite[0]

    # Valid witness for CEGIS-01
    ok_valid, msg_valid = verify_candidate_json(
        t0,
        {
            "approve_tx": True,
            "dual_sig": True,
            "emergency_override": False,
            "escrow_lock": True,
            "sanctions_clear": True,
            "tranche_a": 54000,
            "tranche_b": 50500,
            "tranche_c": 33000,
        },
    )
    assert ok_valid is True
    assert "VERIFIED" in msg_valid

    # Invalid boolean + sum violation
    ok_bad, msg_bad = verify_candidate_json(
        t0,
        {
            "approve_tx": True,
            "dual_sig": False,
            "emergency_override": True,
            "escrow_lock": True,
            "sanctions_clear": True,
            "tranche_a": 60000,
            "tranche_b": 50000,
            "tranche_c": 30000,
        },
    )
    assert ok_bad is False
    assert "CDCL_UNSAT" in msg_bad
    assert "QF_LRA_BOUND_VIOLATION" in msg_bad
    assert "QF_LRA_SUM_VIOLATION" in msg_bad

    # Schema error cases
    assert verify_candidate_json(t0, {"approve_tx": "yes"})[0] is False
    assert (
        verify_candidate_json(
            t0,
            {
                "approve_tx": True,
                "dual_sig": True,
                "emergency_override": False,
                "escrow_lock": True,
                "sanctions_clear": True,
            },
        )[0]
        is False
    )
    assert (
        verify_candidate_json(
            t0,
            {
                "approve_tx": True,
                "dual_sig": True,
                "emergency_override": False,
                "escrow_lock": True,
                "sanctions_clear": True,
                "tranche_a": True,
                "tranche_b": 50000,
                "tranche_c": 30000,
            },
        )[0]
        is False
    )
    assert (
        verify_candidate_json(
            t0,
            {
                "approve_tx": True,
                "dual_sig": True,
                "emergency_override": False,
                "escrow_lock": True,
                "sanctions_clear": True,
                "tranche_a": "not-a-number",
                "tranche_b": 50000,
                "tranche_c": 30000,
            },
        )[0]
        is False
    )

    # Force offline replay against unreachable port using saved benchmarks/real_qwen_eval.json
    rep = run_real_ollama_benchmark(
        base_url="http://127.0.0.1:59999",
        allow_offline_replay=True,
    )
    assert rep.total_tasks == 10
    assert rep.axiom_verified_pass_count >= rep.raw_pass_at_1_count
    assert rep.axiom_formal_violation_pct == 0.0

    with pytest.raises(RuntimeError, match="Ollama daemon unreachable"):
        run_real_ollama_benchmark(
            base_url="http://127.0.0.1:59999",
            save_path=tmp_path / "missing.json",
            allow_offline_replay=False,
        )

    # Also test live CEGIS loop deterministically via mocked httpx transport
    call_counter = {"n": 0}

    def _fake_get(url: str, timeout: float = 2.0) -> httpx.Response:
        return httpx.Response(200, json={"models": []}, request=httpx.Request("GET", url))

    def _fake_post(url: str, json: dict[str, Any], timeout: float = 45.0) -> httpx.Response:
        call_counter["n"] += 1
        req = httpx.Request("POST", url)
        prompt_str = str(json.get("prompt", ""))
        if "FORMAL VERIFIER COUNTEREXAMPLE" not in prompt_str:
            bad_obj = {
                "approve_tx": True,
                "dual_sig": False,
                "emergency_override": True,
                "escrow_lock": False,
                "sanctions_clear": False,
                "tranche_a": 0,
                "tranche_b": 0,
                "tranche_c": 0,
            }
            return httpx.Response(200, json={"response": json_dumps(bad_obj)}, request=req)
        return httpx.Response(200, json={"response": "not-valid-json"}, request=req)

    def json_dumps(obj: dict[str, Any]) -> str:
        return json.dumps(obj)

    monkeypatch.setattr(httpx, "get", _fake_get)
    monkeypatch.setattr(httpx, "post", _fake_post)
    out_file = tmp_path / "simulated_cegis.json"
    sim_rep = run_real_ollama_benchmark(max_attempts=2, save_path=out_file)
    assert sim_rep.total_tasks == 10
    assert out_file.exists()


@pytest.mark.asyncio
async def test_fastapi_workbench_endpoints() -> None:
    fastapi_app = create_app()
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r_health = await client.get("/healthz")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "ok"

        r_root = await client.get("/")
        assert r_root.status_code == 200

        r_ov = await client.get("/api/v1/overview")
        assert r_ov.status_code == 200
        ov = r_ov.json()
        assert ov["speculative"]["forward_pass_reduction_x"] > 1.5
        assert ov["sae"]["reconstruction_r2"] > 0.90
        assert ov["cdcl_demo"]["certificate_verified"] is True

        r_spec = await client.post(
            "/api/v1/speculative/benchmark",
            json={"generated_tokens": 32, "lookahead_k": 3},
        )
        assert r_spec.status_code == 200
        assert r_spec.json()["generated_tokens"] == 32

        r_steer = await client.post(
            "/api/v1/sae/steer",
            json={"steering_alpha": 45.0, "target_regime_id": 3},
        )
        assert r_steer.status_code == 200
        assert r_steer.json()["steered_regime_prob"] > 0.90

        r_prove = await client.post(
            "/api/v1/cdcl/prove",
            json={"num_vars": 2, "clauses": [[1, 2], [-1, 2], [1, -2], [-1, -2]]},
        )
        assert r_prove.status_code == 200
        assert r_prove.json()["status"] == "UNSAT"

        r_mcts = await client.post(
            "/api/v1/mcts/search",
            json={"num_simulations": 24, "c_puct": 1.4},
        )
        assert r_mcts.status_code == 200
        assert r_mcts.json()["verified_terminal_found"] is True


def test_cli_commands_rich_and_json(monkeypatch: pytest.MonkeyPatch) -> None:
    r1 = runner.invoke(cli_app, ["benchmark-speculative", "--tokens", "32", "--json"])
    assert r1.exit_code == 0
    assert json.loads(r1.stdout)["generated_tokens"] == 32

    r2 = runner.invoke(cli_app, ["steer-sae", "--alpha", "40", "--json"])
    assert r2.exit_code == 0
    assert json.loads(r2.stdout)["reconstruction_r2"] > 0.90

    r3 = runner.invoke(cli_app, ["prove", "--json"])
    assert r3.exit_code == 0
    assert json.loads(r3.stdout)["cdcl_certificate"]["status"] == "UNSAT"

    # Force offline replay so CLI test runs in < 50ms even if Ollama is busy/absent
    monkeypatch.setattr(
        "axiom_zero.real_llm_eval.httpx.get",
        lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("offline")),
    )
    r4 = runner.invoke(cli_app, ["eval-ollama", "--offline-replay"])
    assert r4.exit_code == 0
    assert "Real LLM Summary" in r4.stdout

    r5 = runner.invoke(cli_app, ["demo"])
    assert r5.exit_code == 0
    assert "Certificate Verified" in r5.stdout
    assert "Causal Intervention Summary" in r5.stdout
