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
    
    imageTitle.textContent = `${groupKey} [${currentGroupIndex + 1}/${groupKeys.length}]`;
    
    viewer.world.removeAll();
    
    viewer.addTiledImage({
        tileSource: {
            type: 'image',
            url: `/${dataset}/images` + group.center
        },
        x: 0,
        y: 0,
        width: 1
    });
    
    viewer.addTiledImage({
        tileSource: {
            type: 'image',
            url: `/${dataset}/images` + group.position
        },
        x: 1,
        y: 0,
        width: 1
    });
    
    viewer.addTiledImage({
        tileSource: {
            type: 'image',
            url: `/${dataset}/images` + group.az
        },
        x: 2,
        y: 0,
        height: 0.5
    });
    
    viewer.addTiledImage({
        tileSource: {
            type: 'image',
            url: `/${dataset}/images` + group.picks
        },
        x: 2,
        y: 0.6,
        height: 0.5
    });

    viewer.addHandler('tile-loaded', function() {
        viewer.viewport.fitBounds(viewer.world.getHomeBounds());
    });

    const url = new URL(window.location);
    url.searchParams.set('tomogram', groupKey);
    window.history.pushState({}, '', url);

    prefetchNextTomograms();
}

function prefetchNextTomograms() {
    for (let i = 1; i <= 2; i++) {
        if (currentGroupIndex + i < groupKeys.length) {
            const nextGroupKey = groupKeys[currentGroupIndex + i];
            const nextGroup = imageGroups[nextGroupKey];
            
            const imgCenter = new Image();
            imgCenter.src = `/${dataset}/images` + nextGroup.center;
            
            const imgPosition = new Image();
            imgPosition.src = `/${dataset}/images` + nextGroup.position;
            
            const imgAz = new Image();
            imgAz.src = `/${dataset}/images` + nextGroup.az;
            
            const imgPicks = new Image();
            imgPicks.src = `/${dataset}/images` + nextGroup.picks;
        }
    }
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

fetch(`/${dataset}/images`)
    .then(response => response.json())
    .then(data => {
        imageGroups = data;
        groupKeys = Object.keys(imageGroups);
        
        const urlParams = new URLSearchParams(window.location.search);
        const tomogram = urlParams.get('tomogram');
        if (tomogram && groupKeys.includes(tomogram)) {
            currentGroupIndex = groupKeys.indexOf(tomogram);
        }
        
        updateImages();
    });
