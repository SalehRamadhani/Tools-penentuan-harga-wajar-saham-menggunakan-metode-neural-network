
// script untuk input data hanya angka
// Fungsi ini akan menerima hanya angka pada input
function hanyaAngka(evt) {
  var charCode = (evt.which) ? evt.which : evt.keyCode;
  return charCode >= 48 && charCode <= 57;
}

// script untuk mata uang rupiah
function formatCurrency(input) {
  let value = input.value.replace(/\D/g, '');
    
  if (value) {
      let formatted = parseInt(value, 10).toLocaleString('id-ID', {
          style: 'currency',
          currency: 'IDR',
          minimumFractionDigits: 0
      });
      formatted = formatted.replace(/^IDR/, 'Rp');
      input.value = formatted;
  } else {
      input.value = '';
  }
}

function hanyaAngka(event) {
  return /[0-9,]/.test(event.key);
}

// script untuk penulisan angka yang benar
function formatNumber(value) {
  return parseFloat(value).toLocaleString('id-ID', {
    style: 'currency',
    currency: 'IDR',
    minimumFractionDigits: 2
  }).replace(/^IDR/, 'Rp');
}

function cleanFormattedNumber(value) {
  return value.replace(/[^\d]/g, '');
}

//script unuk validasi
let inputSupport = document.getElementById('inputSupport');
let inputResistance = document.getElementById('inputResistance');
let inputHarga = document.getElementById('inputHarga');
let processButton = document.getElementById('processButton');
let inputPBV = document.getElementById('inputPBV');
let inputPER = document.getElementById('inputPER');
let inputEPS = document.getElementById('inputEPS');

function onProcess() {
  // Memformat dan menyimpan data ke localStorage
  localStorage.setItem("Support", cleanFormattedNumber(inputSupport.value));
  localStorage.setItem("Resistance", cleanFormattedNumber(inputResistance.value));
  localStorage.setItem("Harga", cleanFormattedNumber(inputHarga.value));
  localStorage.setItem("PBV", cleanFormattedNumber(inputPBV.value));
  localStorage.setItem("PER", cleanFormattedNumber(inputPER.value));
  localStorage.setItem("EPS", cleanFormattedNumber(inputEPS.value));
  updateRecommended();
  // Mengambil persentase custom dari pengguna
  const userSLPercent = parseFloat(document.getElementById('persenUserSL').value) || 0;
  const userTPPercent = parseFloat(document.getElementById('persenUserTP').value) || 0;
  updateUserValues(userSLPercent, userTPPercent);
}

function updateRecommended(){
  const hargaMasuk = parseFloat(cleanFormattedNumber(localStorage.getItem("Harga")) || "0");

  const recommendedTradePoint = calculateRecommendedTpSl(hargaMasuk);
  localStorage.setItem("takeProfit", recommendedTradePoint.takeProfit.toFixed(2));
  localStorage.setItem("stopLoss", recommendedTradePoint.stopLoss.toFixed(2));

  document.getElementById('SL').innerHTML = 'Recommend Stop Loss : Rp' + localStorage.getItem("stopLoss");
  document.getElementById('TP').innerHTML = 'Recommend Take Profit : Rp' + localStorage.getItem("takeProfit");
}

// proses matematika recoommend
function calculateRecommendedTpSl(hargaMasuk){
  const tpPersen = 35;
  const slPersen = 15;

  const takeProfit = hargaMasuk * (1 + tpPersen / 100);
  const stopLoss = hargaMasuk * (1 - slPersen / 100);
  return{takeProfit,stopLoss};
  
}

// proses matematika custom
function calculateUserTpSl(hargaMasuk, userSLPercent, userTPPercent) {
  const userTakeProfit =  hargaMasuk * (1 + userTPPercent / 100);
  const userStopLoss = hargaMasuk * (1 - userSLPercent / 100);
  return {
    userTakeProfit,userStopLoss
  };
}

// fungsi untuk logika kondisi pada id=keputusan
// Fungsi untuk memperbarui keputusan
function updateDecision() {
  const hargaMasuk = parseFloat(cleanFormattedNumber(localStorage.getItem("Harga")) || "0");
  const hargaWajar = parseFloat(localStorage.getItem("HargaWajar") || "0");
  const support = parseFloat(cleanFormattedNumber(localStorage.getItem("Support")) || "0");
  const resistance = parseFloat(cleanFormattedNumber(localStorage.getItem("Resistance")) || "0");

  let decision = '';

  if (hargaMasuk < hargaWajar && hargaMasuk < support) {
    decision = "Disarankan Beli Bertahap";
  } else if (hargaMasuk < hargaWajar && hargaMasuk > support){
    decision = "Disarankan Beli Bertahap";
  }else if (hargaMasuk > hargaWajar && hargaMasuk < resistance) {
    decision = "Tunggu dan Amati Saham ini";
  } else if (hargaMasuk > hargaWajar && hargaMasuk > resistance) {
    decision = "Disarankan untuk Melewatkan Saham ini";
  } else {
    decision = "Tidak ada saran untuk kondisi saat ini";
  }
  document.getElementById('Keputusan').innerHTML = decision;

  // menampilkan harga wajar saham ke html
  document.getElementById('FV').innerHTML = 'Harga Wajar Saham / Fair Value : Rp' + hargaWajar.toFixed(2);;
}

document.addEventListener('DOMContentLoaded', async function() {
  // validasi
  const inputUserSL = document.getElementById('persenUserSL');
  const inputUserTP = document.getElementById('persenUserTP');
  const userSL = document.getElementById('userSL');
  const userTP = document.getElementById('userTP');

  inputUserSL.addEventListener('input', updateCustomValues);
  inputUserTP.addEventListener('input', updateCustomValues);

  function validateInput(input, max) {
    const currentValue = parseInt(input.value, 10);
    if (currentValue > max) {
      input.value = max; 
    }
  }

  // validasi maksimal input SL adalah 100
  inputUserSL.addEventListener('input', function() {
    validateInput(inputUserSL, 100); 
    updateCustomValues(); 
  });

  // validasi maksimal input TP adalah 200
  inputUserTP.addEventListener('input', function() {
    validateInput(inputUserTP, 200); 
    updateCustomValues(); 
  });

  // Fungsi yang dipanggil ketika input SL atau TP berubah
  function updateCustomValues() {
    const userSLPercent = parseFloat(inputUserSL.value) || 15; 
    const userTPPercent = parseFloat(inputUserTP.value) || 35;
    const hargaMasuk = parseFloat(localStorage.getItem("Harga")) || 0; 

    if (userSLPercent > 0 && userTPPercent > 0 && hargaMasuk > 0) {
      const userTradePoint = calculateUserTpSl(hargaMasuk, userSLPercent, userTPPercent);
      localStorage.setItem("userTakeProfit", userTradePoint.userTakeProfit.toFixed(2));
      localStorage.setItem("userStopLoss", userTradePoint.userStopLoss.toFixed(2));
      document.getElementById('userSL').innerHTML = 'Stop Loss Personalisasi : Rp' + localStorage.getItem("userStopLoss");
      document.getElementById('userTP').innerHTML = 'Take Profit Personalisasi : Rp' + localStorage.getItem("userTakeProfit");
    }
    else{
      document.getElementById('userSL').innerHTML = 'Stop Loss Personalisasi : Rp -';
      document.getElementById('userTP').innerHTML = 'Take Profit Personalisasi : Rp -';
    }

    if (userSLPercent === 15 && userTPPercent === 35){
      userSL.style.display="none";
      userTP.style.display="none";
    } else {
      userSL.style.display="block";
      userTP.style.display="block";
    }
  }

  updateCustomValues();
  inputUserSL.addEventListener('input', updateCustomValues);
  inputUserTP.addEventListener('input', updateCustomValues);

  validateInput(inputUserSL, 100);
  validateInput(inputUserTP, 200);
  updateCustomValues();

  // menampilkan hasil akhir atau mengambil data yang sudah
  // disave pada local storage atau sudah diproses, dimunculkan pada website
  document.getElementById('resis').innerHTML = 'Area Resistance Saham : Rp' + localStorage.getItem("Resistance");
  document.getElementById('supp').innerHTML = 'Area Support Saham : Rp' + localStorage.getItem("Support");
  document.getElementById('SL').innerHTML = 'Recommend Stop Loss : Rp' + localStorage.getItem("stopLoss");
  document.getElementById('TP').innerHTML = 'Recommend Take Profit : Rp' + localStorage.getItem("takeProfit");
  updateDecision(); 
});


