const imageInput = document.getElementById("imageInput");
const detectBtn = document.getElementById("detectBtn");
const result    = document.getElementById("result");
const statusText= document.getElementById("status");

detectBtn.onclick = async () => {
  if (!imageInput.files || imageInput.files.length === 0) {
    alert("Please select an image");
    return;
  }

  statusText.textContent = "Detecting plate...";
  result.textContent = "";

  const formData = new FormData();
  formData.append("image", imageInput.files[0]);

  try {
    // use relative path so it works both locally and on Render
    const response = await fetch("/detect", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (data.count === 0) {
      result.textContent = "No plate detected";
    } else {
      result.textContent = JSON.stringify(data, null, 2);
    }

    statusText.textContent = "Done ✅";
  } catch (err) {
    statusText.textContent = "Server error ❌";
    console.error(err);
  }
};