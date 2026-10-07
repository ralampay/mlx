import subprocess

import pytest

from mlx.core.artifacts import sha256_file
from mlx.core.exceptions import MLXUserError
from mlx.core.repository_snapshot import SnapshotRepositories


def test_snapshot_captures_dirty_and_untracked_files_and_refuses_overwrite(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    subprocess.run(['git','init',str(repo)],check=True,capture_output=True)
    (repo/'code.py').write_text('original\n')
    subprocess.run(['git','-C',str(repo),'add','code.py'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-m','fixture'],check=True,capture_output=True)
    (repo/'code.py').write_text('dirty\n')
    (repo/'new.py').write_text('new\n')
    out=tmp_path/'source'
    metadata=SnapshotRepositories({'test':repo},out).execute()['test']
    assert metadata['files']['code.py']==sha256_file(repo/'code.py')
    assert metadata['files']['new.py']==sha256_file(repo/'new.py')
    assert (out/'test/code.py').read_text()=='dirty\n'
    with pytest.raises(MLXUserError,match='already exists'):
        SnapshotRepositories({'test':repo},out).execute()
