# mlchem - cheminformatics library
# Copyright © 2025 as Unilever Global IP Limited

# Redistribution and use in source and binary forms, with or without modification,
# are permitted under the terms of the BSD-3 License, provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in
#        the documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# You should have received a copy of the BSD-3 License along with mlchem.
# If not, see https://interoperable-europe.ec.europa.eu/licence/bsd-3-clause-new-or-revised-license .
# It is the responsibility of mlchem users to familiarise themselves with all dependencies and their associated licenses.

from pathlib import Path
from setuptools import setup, find_packages

BASE_DIR = Path(__file__).resolve().parent
requirements_path = BASE_DIR / 'requirements.txt'
readme_path = BASE_DIR / 'README.md'
if not requirements_path.exists():
    raise FileNotFoundError(
        "requirements.txt is required for packaging but was not found. "
        "Ensure it is included in the source distribution."
    )

# Read install requirements from requirements.txt
requirements = [
    line.strip()
    for line in requirements_path.read_text(encoding='utf-8').splitlines()
    if line.strip() and not line.strip().startswith('#')
]
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ''

# The repo is laid out flat: the workspace root *is* the `mlchem` package,
# with `chem/`, `ml/`, etc. as subpackages. Tell setuptools that explicitly.
_subpkgs = find_packages(exclude=['tests', 'tests.*', 'docs', 'docs.*'])

setup(
    name='mlchem-ul',
    version='1.1.4',
    description='A Python cheminformatics toolkit for molecular analysis and machine learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/seacunilever/mlchem',
    project_urls={
        'Documentation': 'https://seacunilever.github.io/mlchem/',
        'Source': 'https://github.com/seacunilever/mlchem',
        'Issues': 'https://github.com/seacunilever/mlchem/issues',
    },
    license='BSD-3-Clause',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',

    ],
    package_dir={'mlchem': '.'},
    packages=['mlchem'] + [f'mlchem.{p}' for p in _subpkgs],
    install_requires=requirements,
    python_requires='>=3.12',
)