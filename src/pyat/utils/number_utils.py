#! /usr/bin/env python3
# coding: utf-8


def normalize_longitude(longitude: float) -> float:
    """
    Normalize a longitude [-180, 180]
    """
    longitude = longitude % 360.0
    return longitude - 360.0 if longitude > 180.0 else 360.0 + longitude if longitude < -180.0 else longitude


def normalize_latitude(latitude: float) -> float:
    """
    Normalize a latitude [-90, 90]
    """
    latitude = latitude % 180.0
    return 180.0 - latitude if latitude > 90.0 else -180.0 - latitude if latitude < -90.0 else latitude
