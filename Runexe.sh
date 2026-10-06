#!/bin/bash
sudo python3 -m cProfile -s cumulative hw2.py | grep ForwardTime
