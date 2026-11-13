"""make_model_pack: policy and YOLO packs verify by re-hashing; tampering, extra or missing files, a non-empty destination,
a missing limits file and a bad export are all refused."""

from __future__ import annotations

import json

import pytest

import backends
import make_model_pack as mmp
from export_yolo_onnx import LICENCE_NOTE


@pytest.fixture
def limits(tmp_path):
    p = tmp_path / "g1_limits.yaml"
    p.write_text("validated_on_robot: false\npos_margin_rad: 0.05\n")
    return p


def test_policy_pack_is_written_and_verified(tmp_path, policy_export, limits):
    pack = tmp_path / "pack"
    res = mmp.make_model_pack(policy_export(), pack, note="unit test", limits=limits)
    assert res["kind"] == "policy"
    assert sorted(p.name for p in pack.iterdir()) == [
        "SHA256SUMS", "closedloop.npz", "export.json", "model-card.md", "policy.onnx"]
    assert mmp.verify_pack_dir(pack) == []
    card = (pack / "model-card.md").read_text()
    assert f"onnx_sha256: `{res['onnx_sha256']}`" in card
    assert "git HEAD | not recorded" in card
    assert "unit test" in card and "limits file" in card and backends.sha256(limits) in card


def test_tampered_byte_fails_verification(tmp_path, policy_export, limits):
    pack = tmp_path / "pack"
    mmp.make_model_pack(policy_export(), pack, limits=limits)
    target = pack / "policy.onnx"
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0x01
    target.write_bytes(bytes(data))
    assert any("policy.onnx: sha256 mismatch" in e for e in mmp.verify_pack_dir(pack))


def test_extra_and_missing_files_fail_verification(tmp_path, policy_export, limits):
    pack = tmp_path / "pack"
    mmp.make_model_pack(policy_export(), pack, limits=limits)
    (pack / "stray.bin").write_bytes(b"x")
    (pack / "closedloop.npz").unlink()
    errors = mmp.verify_pack_dir(pack)
    assert "stray.bin is not listed in SHA256SUMS" in errors
    assert "closedloop.npz is listed in SHA256SUMS but missing" in errors


def test_non_empty_destination_is_refused_and_untouched(tmp_path, policy_export, limits):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="not an empty directory"):
        mmp.make_model_pack(policy_export(), pack, limits=limits)
    assert sorted(p.name for p in pack.iterdir()) == ["keep.txt"]


def test_empty_existing_destination_is_accepted(tmp_path, policy_export, limits):
    pack = tmp_path / "pack"
    pack.mkdir()
    mmp.make_model_pack(policy_export(), pack, limits=limits)
    assert mmp.verify_pack_dir(pack) == []


def test_missing_limits_file_is_refused(tmp_path, policy_export):
    pack = tmp_path / "pack"
    with pytest.raises(FileNotFoundError, match="limits file not found"):
        mmp.make_model_pack(policy_export(), pack, limits=tmp_path / "absent.yaml")
    assert not pack.exists()


def test_bad_export_writes_nothing(tmp_path, policy_export, corrupt_onnx, limits):
    export = policy_export()
    corrupt_onnx(export / "policy.onnx", "nan")
    pack = tmp_path / "pack"
    with pytest.raises(ValueError, match="fails check_onnx"):
        mmp.make_model_pack(export, pack, limits=limits)
    assert not pack.exists()


def test_yolo_pack_is_written_and_verified(tmp_path, yolo_onnx):
    export = tmp_path / "yolo_export"
    export.mkdir()
    onnx_path = yolo_onnx(export / "model.onnx", imgsz=64, nc=3)
    (export / "model.json").write_text(json.dumps({
        "names": ["person", "bottle", "cup"], "imgsz": 64, "nc": 3, "opset": 17, "half": False, "simplify": True,
        "sha256": backends.sha256(onnx_path), "source_weights": "yolo11n-seg.pt", "source_weights_sha256": "0" * 64,
        "ultralytics": "stub", "licence_note": LICENCE_NOTE}))
    pack = tmp_path / "yolo_pack"
    res = mmp.make_model_pack(export, pack)
    assert res["kind"] == "yolo"
    assert sorted(p.name for p in pack.iterdir()) == ["SHA256SUMS", "model-card.md", "model.json", "model.onnx"]
    assert mmp.verify_pack_dir(pack) == []
    card = (pack / "model-card.md").read_text()
    assert "AGPL-3.0" in card and "nc 3: person, bottle, cup" in card


def test_yolo_pack_refuses_stale_metadata(tmp_path, yolo_onnx):
    export = tmp_path / "yolo_export"
    export.mkdir()
    yolo_onnx(export / "model.onnx", imgsz=64, nc=3)
    (export / "model.json").write_text(json.dumps({"names": ["a", "b", "c"], "nc": 3, "sha256": "0" * 64}))
    with pytest.raises(ValueError, match="sha256 differs"):
        mmp.make_model_pack(export, tmp_path / "pack")
