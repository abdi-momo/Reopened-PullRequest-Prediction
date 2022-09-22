function Upload() {
    //alert("Hi");
    var fileUpload = document.getElementById("fileUpload");
    var regex = /^([a-zA-Z0-9\s_\\.\-:])+(.csv|.txt)$/;
    if (regex.test(fileUpload.value.toLowerCase())) {
        //if th
        if (typeof (FileReader) != "undefined") {
            var reader = new FileReader();
            reader.onload = function (e) {
            var table = document.createElement('table');
            var rows = e.target.result.split("\n");
            for (var i = 0; i < rows.length; i++) {
                var cells = rows[i].split(",");
                //This is new line of code added
                // table+='<tr>';
                if (cells.length > 1) {
                    var row = table.insertRow(-1);
                    //Inserting table rows
                    for (var j = 0; j < cells.length; j++) {
                        var cell = row.insertCell(-1);
                        //New line of  code
                        //table+='<th>'+row[i]+'</th>';
                        cell.innerHTML = cells[j];
                        }
                        //table='</table>';
                }
                    }
                    var dvCSV = document.getElementById("dvCSV");
                    dvCSV.innerHTML = "";
                    dvCSV.appendChild(table);
                    table.setAttribute("border", "1px solid #000");
                    }
                reader.readAsText(fileUpload.files[0]);
                    } 
            else {
                alert("This browser does not support HTML5.");
            }

            } 
                else {
                    alert("Please upload a valid CSV file.");
                    }
}