if ![ -d "prism_files" ]; then
    pip install gdown
    gdown --id 1X_3UERGPepC76MI060_VJkJXiEkA5MI3 --output output.tar
    docker load --input output.tar
fi
# Create Project folder if not exists
mkdir projects
# Create PRISM file folder if not exists
mkdir prism_files
# Get essential PRISM files
cp ../prism_files/* prism_files
# Run the container
docker run -v "$(pwd)/projects":/projects -v "$(pwd)/prism_files":/prism_files -it coolmc1 bash