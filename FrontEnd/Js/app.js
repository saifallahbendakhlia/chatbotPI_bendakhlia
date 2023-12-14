var BTN=document.querySelector("button")
var TEXTAREA=document.querySelector("textarea")
var DIV=document.querySelector("#reponse_msg")

//EVENEMENT
BTN.addEventListener("click", chatBot)

//fonction principale
function chatBot(){
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
            console.log(data)
        })
    })
    .catch(e=>{
        console.warn(e)
    })


}
