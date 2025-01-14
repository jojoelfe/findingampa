let imageGroups = {};
let currentGroupIndex = 0;
let groupKeys = [];

const imageTitle = document.getElementById('image-title');
const prevButton = document.getElementById('prev-button');
const nextButton = document.getElementById('next-button');

let viewer = OpenSeadragon({
    id: "openseadragon-viewer",
    prefixUrl: "https://openseadragon.github.io/openseadragon/images/",
    tileSources: []
});

function updateImages() {
    const groupKey = groupKeys[currentGroupIndex];
    const group = imageGroups[groupKey];
    
    imageTitle.textContent = groupKey;
    
    viewer.world.removeAll();
    
    viewer.addTiledImage({
        tileSource: {
            type: 'image',
            url: 'images' + group.center
        },
        x: 0,
        y: 0,
        width: 1
    });
    
    viewer.addTiledImage({
        tileSource: {
            type: 'image',
            url: 'images' + group.position
        },
        x: 1,
        y: 0,
        width: 1
    });
    
    viewer.addTiledImage({
        tileSource: {
            type: 'image',
            url: 'images' + group.az
        },
        x: 2,
        y: 0,
        height: 0.5
    });
    
    viewer.addTiledImage({
        tileSource: {
            type: 'image',
            url: 'images' + group.picks
        },
        x: 2,
        y: 0.6,
        height: 0.5
    });

    viewer.addHandler('tile-loaded', function() {
        viewer.viewport.fitBounds(viewer.world.getHomeBounds());
    });
}

prevButton.addEventListener('click', () => {
    if (currentGroupIndex > 0) {
        currentGroupIndex--;
        updateImages();
    }
});

nextButton.addEventListener('click', () => {
    if (currentGroupIndex < groupKeys.length - 1) {
        currentGroupIndex++;
        updateImages();
    }
});

fetch('images')
    .then(response => response.json())
    .then(data => {
        imageGroups = data;
        groupKeys = Object.keys(imageGroups);
        updateImages();
    });
