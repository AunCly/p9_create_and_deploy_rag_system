import os

from script.downloader import download_data

class TestDownloader:

    def test_download_data(self):
        json_path = "data/raw/events.json"
        if os.path.exists(json_path):
            os.remove(json_path)

        download_data()

        assert os.path.exists(json_path)
