from paderbox.io.download import download_file_list
from sacred import Experiment


ex = Experiment()


@ex.config
def config():
    database_path = None
    msg = ('You have to specify where the LibriSpeech database '
           '(only test_clean) should be stored.')
    assert database_path is not None, msg
    database = 'async WASN'
    databases = ['async WASN', 'librispeech']
    msg = 'database must be "async WASN" either or "librispeech".'
    assert database in databases, msg
    libri_url = 'http://www.openslr.org/resources/12/test-clean.tar.gz'
    async_wasn_url = \
        'https://zenodo.org/record/5679070/files/async_wasn.tar.gz'


@ex.automain
def download(database, database_path, async_wasn_url, libri_url):
    if database == 'async WASN':
        download_file_list([async_wasn_url], database_path, exist_ok=True)
    elif database == 'librispeech':
        download_file_list([libri_url], database_path, exist_ok=True)
