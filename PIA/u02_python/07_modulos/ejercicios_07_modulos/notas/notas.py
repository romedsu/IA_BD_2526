#!/usr/bin/env python 3

import estudiantes
import promedio

alumno=input("Nombre del alumno: ")

try:
    media=promedio.promedio(alumno,estudiantes.dic)
    print(f"NOTA MEDIA: {media}")
except (AssertionError,TypeError,ValueError) as e:
    print(f"ERROR: {e}")

