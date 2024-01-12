window.addEventListener("load", function () {
  resetUpload();
});

document.getElementById("imageInput").addEventListener("change", function (e) {
  var reader = new FileReader();
  reader.onload = function (event) {
    document.getElementById("uploadedImage").src = event.target.result;
    document.getElementById("uploadedImage").style.display = "block";
    // Center the image
    document.getElementById("uploadedImage").style.margin = "auto";
  };
  reader.readAsDataURL(e.target.files[0]);
  // Enable the upload button when an image is selected
  document.getElementById("uploadButton").disabled = false;
});

document.getElementById("uploadButton").addEventListener("click", function () {
  var imageInput = document.getElementById("imageInput");
  if (!imageInput.files || !imageInput.files[0]) {
    document.getElementById("apiResponse").innerText = "Upload an image first";
    return; // Don't proceed further if no image is selected
  }

  var file = imageInput.files[0];
  var url =
    "https://500pkgt9ge.execute-api.us-east-1.amazonaws.com/production-turtle/predict_turtles";

  fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-image",
    },
    body: file, // Send the file directly as the body of the request
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.text();
    })
    .then((data) => {
      document.getElementById("apiResponse").innerText = data;
    })
    .catch((error) => {
      console.error("Error:", error);
      document.getElementById("apiResponse").innerText =
        "Failed to get a response: " + error.message;
    });
});

// Add event listener for the reset button
document.getElementById("resetButton").addEventListener("click", function () {
  resetUpload();
});

// Define the resetUpload function
function resetUpload() {
  document.getElementById("imageInput").value = "";
  document.getElementById("uploadedImage").style.display = "none";
  document.getElementById("uploadedImage").src = "";
  document.getElementById("apiResponse").innerText = "";
  document.getElementById("uploadButton").disabled = true; // Disable the upload button initially
}

// Predator section of website

// predator data
const predators = [
  {
    text: "Ghost Crab 🦀 - Ghost crabs, notable for their pale color and ability to blend with sandy beaches, are significant predators of sea turtle hatchlings. These crabs, primarily active at night, lie in wait near sea turtle nesting sites. As hatchlings emerge from their nests and clumsily make their way towards the ocean, ghost crabs seize the opportunity to attack. They are agile and quick, able to snatch the vulnerable hatchlings with their claws. The crabs generally target the hatchlings while they are still on the beach, as their mobility and hunting efficiency decrease in water. This predation is a natural part of coastal ecosystems, but it poses a significant challenge for sea turtle populations, many of which are already endangered due to human activities like habitat destruction and climate change. The ghost crab, while a predator, plays an essential role in the beach's ecological balance, acting as a scavenger and aerator of the sand, contributing to the overall health of the shoreline environment.",
    img: "./resources/ghost-crab.png",
  },
  {
    text: "Seagull 🐦 - Seagulls, often seen patrolling shorelines and coastal regions, are opportunistic predators that pose a significant threat to sea turtle hatchlings. As the hatchlings emerge from their nests, usually under the cover of darkness, they instinctively head towards the ocean, a journey fraught with danger. Seagulls, adept at spotting vulnerable prey, swoop down to snatch the young turtles in their beaks. They are particularly attracted to the movement of the hatchlings and their small size, which makes them easy targets. These birds can be relentless, often working in groups to pick off as many hatchlings as they can. While their predation is a natural part of the ecosystem and plays a role in controlling sea turtle populations, the current vulnerability of many turtle species due to human-related factors makes the impact of seagulls more significant. Seagulls, as predators, have adapted well to human environments, and their increasing populations in some areas can lead to higher predation rates on sea turtle hatchlings, further challenging the survival odds of these endangered creatures",
    img: "./resources/seagull.png",
  },
  {
    text: "Shark 🦈 - Sharks, as apex predators of the oceans, have a critical role in maintaining the balance of marine ecosystems. Among their prey are sea turtles, which can become targets especially during their vulnerable juvenile stages or when injured adults venture too close to the shore. While not all species of sharks hunt sea turtles, some, like the tiger shark, have developed specific hunting strategies and powerful jaws equipped with serrated teeth to penetrate the turtles' hard shells. The presence of sharks near shorelines may indicate a healthy, thriving ocean, as they help to regulate the species below them in the food chain, including the population of sea turtles. However, this predatory relationship is finely balanced, and the decline in shark populations due to human activities has profound effects, potentially disrupting this ancient natural order.",
    img: "./resources/shark.png",
  },
];

let currentPredatorIndex = 0;

// Function to update the content
function updatePredatorContent() {
  const predator = predators[currentPredatorIndex];
  document.getElementById("predatorText").innerText = predator.text;
  document.getElementById("predatorImage").src = predator.img;

  currentPredatorIndex = (currentPredatorIndex + 1) % predators.length;
}

// Set interval to update content every 20 seconds
setInterval(updatePredatorContent, 20000);

// Initialize with the first predator's content
window.addEventListener("load", updatePredatorContent);
