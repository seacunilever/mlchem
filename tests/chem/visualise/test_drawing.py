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

import pytest
from unittest.mock import patch
from rdkit import Chem
from PIL import Image
from mlchem.chem.visualise.drawing import MolDrawer
from mlchem.chem.manipulation import create_molecule

@pytest.fixture
def mol_drawer():
    mol = Chem.MolFromSmiles('CCO')
    return MolDrawer(mol=mol, highlightAtoms=[0, 1], size=[400, 400], legend='Ethanol')

def test_init(mol_drawer):
    assert mol_drawer.mol is not None
    assert mol_drawer.highlightAtoms == [0, 1]
    assert mol_drawer.size == [400, 400]
    assert mol_drawer.legend == 'Ethanol'

def test_show_palette(mol_drawer):
    palette = {'C': (0, 0, 0.3), 'O': (0.8, 0.1, 0)}
    
    with patch('matplotlib.pyplot.show') as mock_show:
        mol_drawer.show_palette(palette)
        assert mock_show.called

def test_update_drawing_options(mol_drawer):
    new_options = {'backgroundColour': 'blue', 'highlightColour': 'yellow'}
    mol_drawer.update_drawing_options(**new_options)
    assert mol_drawer.drawing_options['backgroundColour'] == 'blue'
    assert mol_drawer.drawing_options['highlightColour'] == 'yellow'

def test_reset_drawing_options(mol_drawer):
    mol_drawer.drawing_options['backgroundColour'] = 'blue'
    mol_drawer.reset_drawing_options()
    assert mol_drawer.drawing_options['backgroundColour'] == 'white'

def test_load_images(mol_drawer):
    images = [mol_drawer.draw_mol(create_molecule('CCC')),
              mol_drawer.draw_mol(create_molecule('CCO'))]
    mol_drawer.load_images(images)
    assert mol_drawer.img_list == images

def test_load_mols(mol_drawer):
    mols = [Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('CCN')]
    mol_drawer.load_mols(mols)
    assert len(mol_drawer.img_list) == len(mols)


def test_draw_mol(mol_drawer):
    image = mol_drawer.draw_mol()
    assert isinstance(image, Image.Image)

def test_show_images_grid(mol_drawer):
    images = [mol_drawer.draw_mol(create_molecule('CCC')),
              mol_drawer.draw_mol(create_molecule('CCO'))]
    mol_drawer.load_images(images)
    
    with patch('mlchem.chem.visualise.drawing.MolDrawer.show_images_grid') as mock_display:
        mol_drawer.show_images_grid()
        assert mock_display.called

if __name__ == "__main__":
    pytest.main()