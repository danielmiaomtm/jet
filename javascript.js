<button type="button" onclick="alert('Welcome!')"> click here </button>



<!DOCTYPE html>
<html>
<body>

<script>
function changeColor() {
	element = document.getElementById('myimage');
	if (element.src.match("bulbon")) {
		element.src ="/i/eg_bulboff.gif";
	} else {
		element.src="/i/eg_bulbon.gif";
	}
}
</script>


<img id="myimage" onclick="changeColor()" src="/i/eg_bulboff.gif">


</body>
</html>