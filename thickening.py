def thickening(image, max_iter=100):
    # Pastikan gambar adalah biner
    prev = np.zeros(image.shape, np.uint8)

    # Definisikan elemen struktur untuk thickening (rotasi 0°, 90°, 180°, 270°)
    kernels = [
        np.array([[0, 0, 0],
                  [-1, 1, -1],
                  [1, 1, 1]], dtype=np.int8),
        np.array([[-1, 0, 0],
                  [1, 1, 0],
                  [-1, 1, -1]], dtype=np.int8),
        np.array([[1, 1, 1],
                  [-1, 1, -1],
                  [0, 0, 0]], dtype=np.int8),
        np.array([[-1, 1, -1],
                  [0, 1, 1],
                  [0, 0, -1]], dtype=np.int8),
    ]

    for i in range(max_iter):
        for kernel in kernels:
            hitormiss = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
            image = cv2.bitwise_or(image, hitormiss)
        if np.array_equal(image, prev):
            break
        prev = image.copy()
    return image
