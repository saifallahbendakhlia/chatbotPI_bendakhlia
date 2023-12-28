var BTN=document.querySelector("#bText")
var TEXTAREA=document.querySelector("#textSpeech")
var DIV=document.querySelector("#reponse_msg")
var BTN_MIC=document.querySelector("#bMic")
var BTN_SP=document.querySelector("#bSpeack")
//var SpeechRecognition = SpeechRecognition || webkitSpeechRecognition
var recognition = new webkitSpeechRecognition();
recognition.continuous = false;
recognition.lang = 'en-US';
recognition.interimResults = false;
recognition.maxAlternatives = 1;
//EVENEMENT
BTN.addEventListener("click", chatBot)
BTN_MIC.addEventListener("click", speechToText)
//fonction principale
function chatBot(){
    console.log("je suis chatbot")
    
    let text=TEXTAREA.value
    //je dois communiquer avec le backend
    var url_backend="http://127.0.0.1:8000/analyse"
    fetch(url_backend,
        {
            method:"POST",
            body:JSON.stringify({"texte":text}),
            headers:{  
                'Content-Type': 'application/json'
            }          
        })
    .then(reponse=>{
        reponse.json()
        .then(data=>{
            BTN_SP.style.display=""
            BTN_SP.addEventListener("click",textToSpeech(data.msg))
            console.log(data.msg)
            var i = 0;
            var txt = data.msg; /* The text */
            var speed = 50; /* The speed/duration of the effect in milliseconds */

            function typeWriter() {
            if (i < txt.length) {
                DIV.innerHTML += txt.charAt(i);
                i++;
                setTimeout(typeWriter, speed);
            }
            }
            typeWriter()
        })
    })
    .catch(e=>{
        console.warn(e)
    })
}
function textToSpeech(texte){
 //lire sous format Audio  
    let utterance = new SpeechSynthesisUtterance(texte);
    speechSynthesis.speak(utterance); 


}
function speechToText(){
    alert("Je suis speech to text")
    //1ère partie déclencher l'API Speech To Text
    recognition.start();

}


recognition.onresult = function(event) {

    //2ème partie récupérer le texte
    var message = event.results[0][0].transcript;
    console.log('Result received: ' + message + '.');
     console.log('Confidence: ' + event.results[0][0].confidence);

    //3ème partie remplir l'input en utilisant ce texte
    TEXTAREA.value=message
  }