#!/bin/bash
docker exec -itd LookingFront tensorboard --logdir=. --host=0.0.0.0 --port=${@-6006}
