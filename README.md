# analog_gauge_reader

## Setup instructions

Start by cloning the repository into a directory of your choice. We recommend using GitHub repositories via SSH, instead of HTTPS. In case you have not yet generated an SSH-key and/or linked it to GitHub, please follow this short [guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). The repository can then be clone using

```shell
# cd <your chosen project directory>
git clone git@github.com:ethz-asl/analog_gauge_reader.git
```

To maintain a consistent code style and catch some common types of mistakes, `pre-commit` can be used to automatically format and lint the repository's code whenever a new commit is created. In case you are interested, a detailed guide and installation instructions are available [here](https://pre-commit.com/). The tool can be installed with

```shell
pip3 install pre-commit
```

You can then enable it for this project by calling

```shell
# cd <your chosen project directory>
pre-commit install
```

After the above command, `pre-commit` will automatically check all changed files whenever you try to commit them. You can also run it on all of the repository's files manually at any time by calling `pre-commit run --all-files` and add/customize the checks it performs by editing its config file `.pre-commit-config.yaml` located in this repository's root directory.
