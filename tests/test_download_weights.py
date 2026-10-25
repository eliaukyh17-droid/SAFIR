"""Download selection and integrity checks without network access."""
import hashlib
import json
import sys
from pathlib import Path

import pytest
from tools import download_weights


@pytest.mark.parametrize('group,expected', [
    ('fusion', {'weights/SAFIR_final.safetensors'}),
    ('priors', {'pretrained/yolov8s.pt', 'pretrained/segformer-b2/pytorch_model.bin',
                'pretrained/segformer-b2/config.json'}),
    ('evaluators', {'pretrained/evaluators/fmb_final.pth', 'pretrained/evaluators/m3fd_last.pt'}),
    ('all', {'weights/SAFIR_final.safetensors', 'pretrained/yolov8s.pt',
             'pretrained/segformer-b2/pytorch_model.bin', 'pretrained/segformer-b2/config.json',
             'pretrained/evaluators/fmb_final.pth', 'pretrained/evaluators/m3fd_last.pt'}),
])
def test_download_group_selects_exact_assets(group, expected, tmp_path, monkeypatch):
    selected = []
    monkeypatch.setattr(download_weights, 'download',
                        lambda path, url, checksum: selected.append(path.relative_to(tmp_path).as_posix()))
    monkeypatch.setattr(sys, 'argv', ['download_weights', '--root', str(tmp_path), '--only', group])
    download_weights.main()
    assert set(selected) == expected
    assert len(selected) == len(expected)


def test_evaluator_urls_and_checksums_match_manifest():
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'weights/evaluators.json').read_text())
    assets = {name: (url, checksum) for name, url, checksum in download_weights.FILES}
    for record in manifest['checkpoints']:
        assert assets[record['file']] == (record['download_url'], record['sha256'])


def test_download_verifies_content_before_installing(tmp_path):
    source = tmp_path / 'source.bin'
    source.write_bytes(b'verified checkpoint bytes')
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / 'downloaded' / 'model.bin'
    download_weights.download(destination, source.as_uri(), expected)
    assert destination.read_bytes() == source.read_bytes()
    # A correct existing file is reused without contacting its source.
    source.unlink()
    download_weights.download(destination, source.as_uri(), expected)
    destination.write_bytes(b'damaged')
    with pytest.raises(ValueError, match='Existing file checksum mismatch'):
        download_weights.download(destination, source.as_uri(), expected)
    assert destination.read_bytes() == b'damaged'


def test_bad_download_cannot_replace_final_file(tmp_path):
    source = tmp_path / 'source.bin'
    source.write_bytes(b'wrong content')
    destination = tmp_path / 'downloaded.bin'
    with pytest.raises(ValueError, match='Download checksum mismatch'):
        download_weights.download(destination, source.as_uri(), '0' * 64)
    assert not destination.exists()
