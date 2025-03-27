"""
Create a small test dataset for ArtExtract project.
Downloads sample images for different art styles.
"""

import os
import requests
from pathlib import Path
import logging
from PIL import Image
from io import BytesIO
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample images for testing (public domain artworks)
TEST_IMAGES = {
    'impressionism': [
        ('monet_water_lilies.jpg', 'https://upload.wikimedia.org/wikipedia/commons/a/aa/Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg'),
        ('van_gogh_starry_night.jpg', 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'),
        ('renoir_luncheon.jpg', 'https://upload.wikimedia.org/wikipedia/commons/8/8d/Pierre-Auguste_Renoir_-_Luncheon_of_the_Boating_Party_-_Google_Art_Project.jpg'),
        ('monet_impression.jpg', 'https://upload.wikimedia.org/wikipedia/commons/5/59/Monet_-_Impression%2C_Sunrise.jpg'),
        ('van_gogh_cafe.jpg', 'https://upload.wikimedia.org/wikipedia/commons/2/21/Van_Gogh_-_Terrasse_des_Caf%C3%A9s_an_der_Place_du_Forum_in_Arles_am_Abend1.jpeg'),
        ('monet_poppies.jpg', 'https://upload.wikimedia.org/wikipedia/commons/a/a5/Claude_Monet_-_Les_Coquelicots.jpg'),
        ('renoir_moulin.jpg', 'https://upload.wikimedia.org/wikipedia/commons/b/b2/Pierre-Auguste_Renoir_-_Bal_du_moulin_de_la_Galette.jpg'),
        ('sisley_snow.jpg', 'https://upload.wikimedia.org/wikipedia/commons/b/b4/Alfred_Sisley_-_Snow_at_Louveciennes_-_1878.jpg'),
        ('pissarro_garden.jpg', 'https://upload.wikimedia.org/wikipedia/commons/8/8f/Camille_Pissarro_-_Kitchen_Garden_with_Trees_in_Flower%2C_Spring%2C_Pontoise_-_1877.jpg'),
        ('manet_olympia.jpg', 'https://upload.wikimedia.org/wikipedia/commons/5/5c/Edouard_Manet_-_Olympia_-_Google_Art_Project_3.jpg')
    ],
    'baroque': [
        ('rembrandt_night_watch.jpg', 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/The_Night_Watch_-_HD.jpg/1280px-The_Night_Watch_-_HD.jpg'),
        ('vermeer_pearl_earring.jpg', 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/1665_Girl_with_a_Pearl_Earring.jpg/800px-1665_Girl_with_a_Pearl_Earring.jpg'),
        ('rembrandt_anatomy.jpg', 'https://upload.wikimedia.org/wikipedia/commons/4/4d/Rembrandt_-_The_Anatomy_Lesson_of_Dr_Nicolaes_Tulp.jpg'),
        ('vermeer_milkmaid.jpg', 'https://upload.wikimedia.org/wikipedia/commons/2/20/Johannes_Vermeer_-_Het_melkmeisje_-_Google_Art_Project.jpg'),
        ('caravaggio_calling.jpg', 'https://upload.wikimedia.org/wikipedia/commons/4/48/The_Calling_of_Saint_Matthew-Caravaggo_%281599-1600%29.jpg'),
        ('rembrandt_return.jpg', 'https://upload.wikimedia.org/wikipedia/commons/9/98/Rembrandt_-_Return_of_the_Prodigal_Son_-_WGA19135.jpg'),
        ('vermeer_view.jpg', 'https://upload.wikimedia.org/wikipedia/commons/a/a2/Vermeer-view-of-delft.jpg'),
        ('caravaggio_judith.jpg', 'https://upload.wikimedia.org/wikipedia/commons/b/b2/Caravaggio_Judith_Beheading_Holofernes.jpg'),
        ('rubens_raising.jpg', 'https://upload.wikimedia.org/wikipedia/commons/1/13/Rubens_-_The_Raising_of_the_Cross.jpg'),
        ('velazquez_surrender.jpg', 'https://upload.wikimedia.org/wikipedia/commons/e/e8/La_Rendici%C3%B3n_de_Breda.jpg')
    ],
    'renaissance': [
        ('da_vinci_mona_lisa.jpg', 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/687px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg'),
        ('botticelli_venus.jpg', 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg/1280px-Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg'),
        ('raphael_school_athens.jpg', 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/%22The_School_of_Athens%22_by_Raffaello_Sanzio_da_Urbino.jpg/1280px-%22The_School_of_Athens%22_by_Raffaello_Sanzio_da_Urbino.jpg'),
        ('michelangelo_creation.jpg', 'https://upload.wikimedia.org/wikipedia/commons/5/5b/Michelangelo_-_Creation_of_Adam_%28cropped%29.jpg'),
        ('leonardo_vitruvian.jpg', 'https://upload.wikimedia.org/wikipedia/commons/2/22/Da_Vinci_Vitruve_Luc_Viatour.jpg'),
        ('botticelli_spring.jpg', 'https://upload.wikimedia.org/wikipedia/commons/3/3c/Botticelli-primavera.jpg'),
        ('raphael_transfiguration.jpg', 'https://upload.wikimedia.org/wikipedia/commons/d/d7/Transfiguration_Raphael.jpg'),
        ('michelangelo_pieta.jpg', 'https://upload.wikimedia.org/wikipedia/commons/1/1f/Michelangelo%27s_Pieta_5450_cropncleaned_edit.jpg'),
        ('leonardo_last_supper.jpg', 'https://upload.wikimedia.org/wikipedia/commons/4/4b/%C3%9Altima_Cena_-_Da_Vinci_5.jpg'),
        ('botticelli_pallas.jpg', 'https://upload.wikimedia.org/wikipedia/commons/4/42/Pallas_and_the_Centaur_-_Botticelli.jpg')
    ]
}

def download_and_save_image(url: str, save_path: Path, target_size: tuple = (224, 224)) -> bool:
    """
    Download an image from URL and save it to the specified path.
    
    Args:
        url: URL of the image
        save_path: Path to save the image
        target_size: Target size for the image (width, height)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        headers = {
            'User-Agent': 'ArtExtract/1.0 (Educational Project; jonathan.muhire@example.com) Python/3.9'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Open and resize image
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')  # Ensure RGB format
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Save image
        img.save(save_path, 'JPEG', quality=95)
        logger.info(f"Successfully downloaded and saved: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False

def create_test_dataset(base_dir: str = 'data/test_dataset') -> None:
    """
    Create test dataset by downloading sample images.
    
    Args:
        base_dir: Base directory for the test dataset
    """
    base_dir = Path(base_dir)
    metadata = []
    
    for style, images in TEST_IMAGES.items():
        style_dir = base_dir / 'style' / style
        style_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, url in images:
            save_path = style_dir / filename
            if download_and_save_image(url, save_path):
                metadata.append({
                    'filename': str(save_path.relative_to(base_dir)),
                    'style': style,
                    'source_url': url
                })
    
    # Save metadata
    metadata_file = base_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")

if __name__ == "__main__":
    create_test_dataset()
    logger.info("Test dataset creation completed")
