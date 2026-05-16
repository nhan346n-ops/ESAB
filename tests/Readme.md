# Pyat unit tests

This directory contains unit tests.
pytest framework is used and test can be launched with ```pytest .``` under pyat root directory.

## Pyat unit test files

Test files are available through [generic package registry of Pyat](https://gitlab.ifremer.fr/fleet/pyat/-/packages).

### Implement a new test based on pyat test files

To implement a new test based on pyat test files, use the following command in your test :

```
from pyat.test.file_test_installer import get_test_path

test_file_path = get_test_path() / "<subdirectory>" / "<testfilename>"
```

On first call, if it does not already exist, pyat_test_file directory will be downloaded from pyat package registry
and unzipped under pyat/data/external. Data are then kept for further tests, until being manually deleted.

Exemple :

```
from pyat.test.file_test_installer import get_test_path

MBG_PATH = get_test_path() / "mbg" / "0136_20120607_083636_ShipName_ref.mbg"
```

### Retrieve test files

To manualy retrieve the files use the following command in a terminal :

```
curl "https://gitlab.ifremer.fr/api/v4/projects/343/packages/generic/pyat_test_file/0.0.1/pyat_test_file.zip" --output /full/path/to/your/local/pyat_test_file.zip
```

```
pyat_test_file
├── mbg
│   ├── File1.mbg
│   └── File2.mbg
├── nvi
│   ├── File1.nvi
│   └── File2.nvi
└── ...
```

### Add and push new test file

In your local pyat_test_file directory (ex: /s7k/mydata.s7k) :

- create a subdirectory if needed (ex: pyat_test_file/s7k)
- add your file (ex: pyat_test_file/s7k/mydata.s7k)
- zip it
- push a new release with the following command in a terminal, updating the version if needed (0.0.1 for now):
    ```
    curl --user "NAME:PASSWD" --upload-file /full/path/to/your/local/pyat_test_file.zip "https://gitlab.ifremer.fr/api/v4/projects/343/packages/generic/pyat_test_file/0.0.1/pyat_test_file.zip?select=package_file"
    ```

The new pyat_test_file will look like that :

```
pyat_test_file
├── mbg
│   ├── File1.mbg
│   └── File2.mbg
├── nvi
│   ├── File1.nvi
│   └── File2.nvi
├── s7k
│   ├── File1.s7k
│   └── File2.s7k
└── ...
```

