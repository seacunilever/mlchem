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