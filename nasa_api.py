#!/usr/bin/env python
import asyncio
import aiohttp
import pandas as pd


class NasaAPI:
    def __init__(self, parameters, coordinates, date_range):


