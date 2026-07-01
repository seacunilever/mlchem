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
import numpy as np
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


def test_show_images_grid_invalid_empty_tile_colour_raises(mol_drawer):
    with pytest.raises(AssertionError, match='is not a known colour'):
        mol_drawer.show_images_grid(images=[], empty_tile_colour='not_a_colour')


def test_show_images_grid_invalid_size_raises(mol_drawer):
    with pytest.raises(AssertionError, match="'size' argument must have lenght == 2"):
        mol_drawer.show_images_grid(images=[], size=[100])


def test_draw_mol_requires_valid_molecule():
    drawer = MolDrawer(mol=None)
    with pytest.raises(AssertionError, match='No valid molecule was passed'):
        drawer.draw_mol()


def test_draw_mol_invalid_background_colour_string_raises(mol_drawer):
    mol_drawer.update_drawing_options(backgroundColour='not_a_colour')
    with pytest.raises(ValueError, match='not a valid colour'):
        mol_drawer.draw_mol()


def test_draw_mol_invalid_atom_palette_raises_assertion(mol_drawer):
    mol_drawer.update_drawing_options(atomPalette='invalid_palette')
    with pytest.raises(AssertionError, match="'atomPalette' property must be one of"):
        mol_drawer.draw_mol()


def test_draw_mol_invalid_highlight_colour_string_raises(mol_drawer):
    mol_drawer.update_drawing_options(highlightColour='not_a_colour')
    with pytest.raises(ValueError, match='not a valid colour'):
        mol_drawer.draw_mol()


def test_draw_mol_invalid_query_colour_string_raises(mol_drawer):
    mol_drawer.update_drawing_options(queryColour='not_a_colour')
    with pytest.raises(ValueError, match='not a valid colour'):
        mol_drawer.draw_mol()


def test_draw_mol_acs1996_mode_returns_image(mol_drawer):
    img = mol_drawer.draw_mol(ACS1996_mode=True)
    assert isinstance(img, Image.Image)


def test_draw_mol_with_weight_circle_map_style_branch(mol_drawer):
    atom_count = mol_drawer.mol.GetNumAtoms()
    mol_drawer.update_drawing_options(
        atomWeights=[0.5 if i % 2 == 0 else -0.4 for i in range(atom_count)],
        mapStyle='C',
        numContours=2,
    )
    img = mol_drawer.draw_mol()
    assert isinstance(img, Image.Image)


def test_draw_mol_with_custom_circle_shape_branch(mol_drawer):
    mol_drawer.update_drawing_options(
        shapeTypes=['circle'],
        shapeSizes=[0.3],
        shapeColours=['red'],
        shapeCoords=[(0.0, 0.0)],
    )
    img = mol_drawer.draw_mol()
    assert isinstance(img, Image.Image)


def test_draw_mol_tuple_highlight_atoms_is_handled(mol_drawer):
    img = mol_drawer.draw_mol(highlightAtoms=(0, 1))
    assert isinstance(img, Image.Image)


def test_draw_mol_numpy_highlight_atoms_is_handled(mol_drawer):
    img = mol_drawer.draw_mol(highlightAtoms=np.array([0, 1]))
    assert isinstance(img, Image.Image)

if __name__ == "__main__":
    pytest.main()