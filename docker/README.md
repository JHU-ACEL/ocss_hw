# Docker 
For Windows and Mac you will need to install [Docker Desktop](https://docs.docker.com/desktop/)
### Build Docker Image
Run this in the home repo to build the image
```bash
docker compose build
```

**Windows:**

Windows require a few additional steps to build the docker image. Git converts
LF endings to CRLF endings automatically when pulling .sh scripts on a windows system to maintain standard 
as linux/mac uses LF line endings while windows uses CRLF endings. While this can be configured to be turned off in git,
It is not recommended. Hence, we'd manually convert the shell scripts back to LF standard using dos2unix command in wsl.

**Option A:**

Manually convert the .sh files in unix format

1 - Open wsl and first intall dos2unix
```bash
apt install dos2unix
```
2 - Navigate to the "docker" folder inside the directory where you cloned the repository, e.g:
```bash
 cd /mnt/c/github/ocss_hw/docker
```
3 - Navigate inside the setup folder and run the following command
```bash
 cd setup
 dos2unix  install_acados.sh install_tera_renderer.sh
```
**Option B:**

Turn auto crlf off in Git bash. Delete the repository if you've already cloned. Open Git Bash and enter the following command. Clone the repository after executing this
command in your preferred local directory
```bash
 cd setup
 git config --global core.autocrlf false
```
Go to the parent repository directory (ocss_hw) and run the following command to build the docker image
```bash
docker compose build
```

### Running Docker Container
Start a container by running
```bash
docker compose up
```
Press `d` to detach and get back to your command line, and then run
```bash
docker exec -it ocss-cli-1 bash
```
to enter the container command line.

Exit the container by typing `exit` or `Cntrl +P +Q`. End the container by running
```bash
docker compose down
```

You can view actively running containers by running `docker ps`, and view built images with `docker images`.

### Run marimo from inside the container

```bash
### Run marimo (from inside the container):
marimo edit --headless --host 0.0.0.0 --port=8888
```

Now type in `localhost:8888` into your browser and copy/paste the access token to start using the notebook.

We have specified the default port to be `8888`, if you want to change that, just make a `.env` file in the home directory of this repo and include 
`MARIMO_PORT=8887` or whatever port number you prefer.

### Marimo helpful tips
When you open the marimo notebook editor, select the lightning bolt icon in the bottom left and change "On Module Change" to "Lazy" or "Autorun".
This makes it so that when you make a change to a file, it automatically updates in the notebook.
You can view and edit files by opening them from the sidebar on the left.
