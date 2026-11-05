import subprocess
import sys
import zipfile
from pathlib import Path


def test_bundler_includes_reports_but_never_raw_records(tmp_path):
    root=tmp_path/'trial';root.mkdir()
    (root/'cbc_study.json').write_text('{"test":true}')
    (root/'train_records.jsonl.gz').write_bytes(b'private data')
    (root/'cbc_full.pt').write_bytes(b'checkpoint')
    out=tmp_path/'diagnostics.zip'
    p=Path(__file__).resolve().parents[1]/'scripts'/'package_diagnostics.py'
    subprocess.run([sys.executable,str(p),str(root),'--output',str(out)],check=True,capture_output=True)
    with zipfile.ZipFile(out) as f:
        names=f.namelist()
    assert any(name.endswith('cbc_study.json') for name in names)
    assert not any('records' in name or '.pt' in name for name in names)
    assert out.stat().st_size < 50*1024*1024
