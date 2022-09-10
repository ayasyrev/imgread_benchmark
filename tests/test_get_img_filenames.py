from imgread.get_img_filenames import get_img_filenames


test_data_dir = 'tests/test_imgs'


def test_get_image_filenames():
    filenames = get_img_filenames(test_data_dir)
    assert isinstance(filenames, list)
    assert len(filenames) == 2

    filenames = get_img_filenames(test_data_dir, num_samples=1)
    assert len(filenames) == 1

    filenames = get_img_filenames(test_data_dir, num_samples=0)
    assert len(filenames) == 2
