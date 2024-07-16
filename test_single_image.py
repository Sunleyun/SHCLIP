def test_single_image(args):
    # Configuration and model loading (similar to before)
    logger, device, model, preprocess = setup_model_and_logging(args)

    # Load and transform a single image
    img_path = 'path/to/your/image.jpg'  # Specify the path to your test image
    image = Image.open(img_path)
    image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and transfer to the appropriate device

    # Anomaly detection
    image_features = model.encode_image(image)
    text_features = model.get_text_features()
    anomaly_score = torch.sum(image_features * text_features)  # Example scoring mechanism

    # Output the results
    print("Anomaly score:", anomaly_score.item())
    visualizer(img_path, anomaly_score, args.image_size, args.save_path)

if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    test_single_image(args)