const leftInput = document.getElementById('left-input');
const rightInput = document.getElementById('right-input');
const leftPreview = document.getElementById('left-preview');
const rightPreview = document.getElementById('right-preview');
const uploadForm = document.getElementById('upload-form');
const submitBtn = document.getElementById('submit-btn');
const loading = document.getElementById('loading');
const leftUpload = document.getElementById('left-upload');
const rightUpload = document.getElementById('right-upload');

// Funzione per gestire l'anteprima delle immagini
function handleImagePreview(input, preview) {
  const file = input.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = e => {
      preview.innerHTML = `
        <img src="${e.target.result}" alt="Preview">
        <p>${file.name}</p>
      `;
      preview.classList.remove('empty');
    };
    reader.readAsDataURL(file);
  } else {
    preview.innerHTML = 'Nessuna immagine selezionata';
    preview.classList.add('empty');
  }
}

// Event listeners per le anteprime
leftInput.addEventListener('change', () => handleImagePreview(leftInput, leftPreview));
rightInput.addEventListener('change', () => handleImagePreview(rightInput, rightPreview));

// Gestione del submit
uploadForm.addEventListener('submit', e => {
  if (!leftInput.files[0] || !rightInput.files[0]) {
    e.preventDefault();
    alert('Seleziona entrambe le immagini prima di procedere');
    return;
  }
  loading.style.display = 'block';
  submitBtn.disabled = true;
  submitBtn.textContent = '⏳ Elaborando immagini stereo...';
});

// Drag & Drop per sinistra
leftUpload.addEventListener('dragover', e => { e.preventDefault(); leftUpload.classList.add('dragover'); });
leftUpload.addEventListener('dragleave', e => { e.preventDefault(); leftUpload.classList.remove('dragover'); });
leftUpload.addEventListener('drop', e => {
  e.preventDefault();
  leftUpload.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    leftInput.files = e.dataTransfer.files;
    handleImagePreview(leftInput, leftPreview);
  }
});

// Drag & Drop per destra
rightUpload.addEventListener('dragover', e => { e.preventDefault(); rightUpload.classList.add('dragover'); });
rightUpload.addEventListener('dragleave', e => { e.preventDefault(); rightUpload.classList.remove('dragover'); });
rightUpload.addEventListener('drop', e => {
  e.preventDefault();
  rightUpload.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    rightInput.files = e.dataTransfer.files;
    handleImagePreview(rightInput, rightPreview);
  }
});

// Smooth scroll ai risultati (liquid template)

setTimeout(() => {
  document.querySelector('.results').scrollIntoView({ behavior: 'smooth' });
}, 100);

