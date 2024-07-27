/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// [START maps_event_click_latlng]
async function initMap() {
  // Request needed libraries.
  const { Map } = await google.maps.importLibrary("maps");
    const myLatlng = {lat: 43.785104, lng: 7.522557};
  const map = new google.maps.Map(document.getElementById("map"), {
    zoom: 20,
    center: myLatlng,
    mapTypeId: 'satellite',
    tilt: 0,
  });
  // Create the initial InfoWindow.
  // let infoWindow = new google.maps.InfoWindow({
  //   content: JSON.stringify(myLatlng, null, 2),
  //   position: myLatlng,
  // });
  const cityCircle = new google.maps.Circle({
    strokeColor: "#FF0000",
    strokeOpacity: 0.8,
    strokeWeight: 2,
    fillColor: "#FF0000",
    fillOpacity: 0.35,
    map,
    center: myLatlng,
    radius: 0.2,
  });
  document.getElementById("lat").textContent = myLatlng.lat;
  document.getElementById("lng").textContent = myLatlng.lng;

  // cityCircle.open(map);
  // [START maps_event_click_latlng_listener]
  // Configure the click listener.
  map.addListener("click", (mapsMouseEvent) => {
    // // Close the current InfoWindow.
    // infoWindow.close();
    // // Create a new InfoWindow.
    // infoWindow = new google.maps.InfoWindow({
    //   position: mapsMouseEvent.latLng,
    // });
    // infoWindow.setContent(
    //   JSON.stringify(mapsMouseEvent.latLng.toJSON(), null, 2),
    // );
    // infoWindow.open(map);
    cityCircle.setCenter(mapsMouseEvent.latLng);
    // Update the text in the spans on the left with ids equal to lat and lng.
    document.getElementById("lat").textContent = mapsMouseEvent.latLng.lat();
    document.getElementById("lng").textContent = mapsMouseEvent.latLng.lng();
  });
  // [END maps_event_click_latlng_listener]
}

initMap();
// [END maps_event_click_latlng]

// The image names come from an html file in the images folder.
// The image index is used to keep track of the current image.
var imageIndex = 0;

var image_names = [
//////// FILL THE ARRAY WITH THE IMAGES YOU WANT TO LABEL ////////

//////////////////////////////////////////////////////////////////
];
// Set camera image to the first image in the array.
document.getElementById("camera_image").src = image_names[imageIndex];

function nextImage(){

  // When the next image button is clicked the image index is incremented by 1
  // and the image is updated. If the image index is greater than the number of
  // images, the index is set to 0.
  imageIndex++;
  // debug to console
  console.log('imageIndex:')
  console.log(imageIndex);
  if(imageIndex >= image_names.length){
    imageIndex = 0;
  }
  // Update the image on the page with the current image index.
  document.getElementById("camera_image").src = image_names[imageIndex];
};

function saveAndNext(){
  // A new row is added to the table with the image index and the current lat and lng.
  var table = document.getElementById("labelling_table");
  var row = table.insertRow(-1);
  var cell1 = row.insertCell(0);
  var cell2 = row.insertCell(1);
  var cell3 = row.insertCell(2);
  cell1.innerHTML = image_names[imageIndex];
  cell2.innerHTML = document.getElementById("lat").textContent;
  cell3.innerHTML = document.getElementById("lng").textContent;
  // The next image is shown.
  nextImage();
}

function download(){
  // Downloads the labelling_table as a csv file.
  var table = document.getElementById("labelling_table");
  var rows = table.rows;
  var csv = [];
  for(var i = 0; i < rows.length; i++){
    var row = [];
    var cells = rows[i].cells;
    for(var j = 0; j < cells.length; j++){
      row.push(cells[j].innerHTML);
    }
    csv.push(row.join(","));
  }
  downloadCSV(csv.join("\n"), "labelling.csv");
}

function downloadCSV(csv, filename) {
  var csvFile;
  var downloadLink;

  // CSV file
  csvFile = new Blob([csv], {type: "text/csv"});

  // Download link
  downloadLink = document.createElement("a");

  // File name
  downloadLink.download = filename;

  // Create a link to the file
  downloadLink.href = window.URL.createObjectURL(csvFile);

  // Hide download link
  downloadLink.style.display = "none";

  // Add the link to DOM
  document.body.appendChild(downloadLink);

  // Click download link
  downloadLink.click();
}

// Link to the button in the html file.
var button = document.getElementById("next_button");
// Add an event listener to the button that calls the saveAndNext function.
button.addEventListener("click", saveAndNext);

// Link to the button in the html file.
var download_button = document.getElementById("download_button");
// Add an event listener to the button that calls the download function.
download_button.addEventListener("click", download);

// Link to the button in the html file.
var skip_button = document.getElementById("skip_button");
// Add an event listener to the button that calls the skip function.
skip_button.addEventListener("click", nextImage);


// Link to the button in the html file.
var skip_30_button = document.getElementById("skip_30_button");
// Add an event listener to the button that calls the skip function.
skip_30_button.addEventListener("click", function(){
  for(var i = 0; i < 30; i++){
    nextImage();
  }
});
