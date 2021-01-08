document.querySelector(".start-btn").addEventListener('click',e=>{
   var img=document.createElement("img");
   img.classList.add("bg");
   img.setAttribute("src","/video_feed");
   document.getElementById("main").appendChild(img);
});

document.querySelector(".stop-btn").addEventListener('click',e=>{
   window.location.href="/livecapture";
})

document.querySelector(".livecapture").addEventListener('click',e=>{
   window.location.href="/upload";
})

function PreviewImage() {
   var oFReader = new FileReader();
   oFReader.readAsDataURL(document.getElementById("uploadImage").files[0]);

   oFReader.onload = function (oFREvent) {
       document.getElementById("uploadPreview").src = oFREvent.target.result;
   };
};