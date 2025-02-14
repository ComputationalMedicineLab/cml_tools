#!/bin/bash
# Run `source sync.sh` to use this function from anywhere (e.g. from project directories)

sync_to_clio() {
    rsync -avzP --exclude='__pycache__' --exclude='*.pyc' $(pwd) /clio/projects/
}
