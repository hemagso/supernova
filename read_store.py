from supernova.metadata import FeatureStore
from devtools import debug


def main():
    path = "./docs/example"
    fs = FeatureStore.from_folder(path)
    debug(fs)


if __name__ == "__main__":
    main()
