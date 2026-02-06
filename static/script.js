function showPage(page) {
    let content = document.getElementById('content');
    let titles = {
        'preencher-solo': `
            <div style="display: flex; justify-content: space-between;"> <!-- Container Flex -->
                <div class="formulario-container" style="width: 50%; float: left; margin-left: 0; border-radius: 10px; border: 1px solid #ccc;"> <!-- Remover margem à esquerda -->
                    <h2 style="margin-bottom: 40px;">Fill Form</h2> <!-- Aumentar o espaçamento abaixo do título -->
                    <div class="input-container" style="margin-top: 40px;"> <!-- Aumentar o espaçamento acima do input container -->
                        <div style="display: flex; align-items: center; margin-bottom: 20px;"> <!-- Flex para alinhar inputs -->
                            <label for="soil-code" style="margin-right: 10px;">Soil Code:</label>
                            <input type="text" id="soil-code" style="width: 100px;"> <!-- Diminuir a largura -->
                            <label for="num-points" style="margin-left: 20px; margin-right: 10px;">Number of Retention Points:</label>
                            <input type="number" id="num-points" style="width: 100px;" onchange="generateInputRows()"> <!-- Diminuir a largura -->
                        </div>

                        <div id="input-rows" style="margin-top: 40px;"></div> <!-- Aumentar o espaçamento entre as linhas -->
                        <button onclick="createExcel()">Run</button>
                        <button id="stopButton" style="display:none;" onclick="stopProcessing()">Stop</button>
                    </div>
                    <div id="loading-indicator" style="display:none;">Processing...</div> <!-- Loading indicator -->
                    <div class="download-buttons" id="form-download-buttons" style="display:none; margin-top: 20px; position: relative; bottom: 0; left: 0;">
                        <a id="file1" href="#" download style="display:none;">Download your Classified Soil</a>
                        <a id="file2" href="#" download style="display:none;">Download your Plots</a>
                    </div>
                </div>
                
                <div class="formulario-container" style="width: 45%; height: 245px; border-radius: 10px; border: 1px solid #ccc; padding: 20px; margin-left: 40px; overflow: auto;"> <!-- Novo container com dimensões fixas -->
                    <h2 style="margin-bottom: 20px;">Important Notes</h2>
                    <p>The retention data must contain at least 4 points following the criteria:</p>
                    <ul>
                        <li> 1st point (φ): suction equal to 0 cm;</li>
                        <li> 2nd point (θ1): suction between 30 and 80 cm;</li>
                        <li> 3rd point (θ2): suction between 250 and 500 cm;</li>
                        <li> 4th point (θ3): suction between 9000 and 18000 cm.</li>
                    </ul>
                    <p style="margin-top: 20px;">* The suction value must be positive.</p>
                </div>
            </div>
        `,
        'importar-xlsx': `
            <div style="display: flex; justify-content: space-between;">
                <div class="formulario-container" id="importar-dados" style="width: 45%; height: 200px; border-radius: 10px; border: 1px solid #ccc; padding: 20px;"> <!-- Altura reduzida -->
                    <h2>Import Data</h2>
                    <div class="upload-container">
                        <input type="file" id="fileInput" />
                        <button onclick="uploadFile()">Run</button>
                        <button id="stopButtonUpload" style="display:none;" onclick="stopUploadProcessing()">Stop</button>
                    </div>
                    <div id="loading-indicator-upload" style="display:none;">Processing...</div> <!-- Loading indicator -->
                    <div class="file-download-buttons" style="margin-top: 20px;">
                        <a id="file1" href="#" download style="display:none;">Download your Classified Soil</a>
                        <a id="file2" href="#" download style="display:none;">Download your Plots</a>
                    </div>
                </div>
                
                <div class="formulario-container" style="width: 45%; height: 540px; border-radius: 10px; border: 1px solid #ccc; padding: 20px; margin-left: 40px; overflow: auto;"> <!-- Novo container de Instruções com altura aumentada -->
                    <h2 style="margin-bottom: 20px;">Instructions</h2>
                    <p>If the user wishes to enter retention data for a database with more than one soil, they can do so using two data entry formats:</p>
                    <ol>
                        <li>Column data for h(cm) and theta<br><img src="assets/example_column_format.png" alt="Example Column Format" style="max-width: 100%; height: auto;"></li>
                        <li>Row data for h(cm) and theta<br><img src="assets/example_row_format.png" alt="Example Row Format" style="max-width: 100%; height: auto;"></li>
                    </ol>
                    <div style="margin-top: 20px; display: flex; gap: 10px;"> <!-- Flex container para os botões -->
                        <a href="/assets/TemplateExcelColumn.xlsx" download style="background-color: black; color: white; padding: 10px 15px; border-radius: 5px; text-decoration: none;">Download Column Template</a>
                        <a href="/assets/TemplateExcelRow.xlsx" download style="background-color: black; color: white; padding: 10px 15px; border-radius: 5px; text-decoration: none;">Download Row Template</a>
                    </div>
                </div>
            </div>
        `,
        'sobre-nos': `
            <div class="content-wrapper" style="display: flex; flex-direction: column; align-items: flex-start; padding: 20px; margin: 0 20px;">
                <div class="formulario-container" style="width: 100%; max-width: 1500px; border: 1px solid #ccc; border-radius: 10px; padding: 30px;">
                    <h2>About the SPSCS</h2>
                        <p>The characterization of the porous structure of soils is carried out through methods such as computed tomography, magnetic resonance, soil density and porosity measurements, and water retention curves. However, the lack of a standardized methodology with precise and quantitative criteria has hindered the consistent understanding of the characteristics and functions of the porous system across different soils. To overcome these limitations, Ottoni's (2017) thesis introduced the Soil Porous Space Structural Classification System (SPSCS). This system offers a standardized and quantitative approach, based on the volumetric structure of the porous space and using air availability curves to classify soils.</p>
                        <p>The SPSCS website was created with the aim of offering a practical and accessible application of this methodology, allowing users to classify soils simply and accurately using the parameters established by the SPSCS. It provides a tool that facilitates the implementation of the system and ensures the standardization of processes, making the methodology more accessible to the scientific community and professionals in the field.</p>
                        <p>Soil classification is done in two hierarchical levels: Order, which groups soils with similar pore size arrangements, and Sub-Order, which groups soils with similar effective porosity, influencing hydrodynamic processes such as infiltration and drainage. The Aa(s) curve is modeled from the van Genuchten equation, with parameters optimized from porosity measurements in the saturation range and three experimental measurements in specific suction ranges: 30 to 60 cm, 250 to 500 cm, and 9000 to 18000 cm.</p>
                        <p>The van Genuchten equation to describe the water retention curve is given by:</p>
                        <p><strong>θ(s) = θr + (ϕ - θr) [1 + (αs)^n]^-m</strong></p>
                        <p>Where:</p>
                        <ul>
                            <li>θ(s) is the volumetric water content in the soil at a given suction ss,</li>
                            <li>θr is the residual water content,</li>
                            <li>ϕ is the total porosity of the soil,</li>
                            <li>α, n, and m are model parameters.</li>
                        </ul>
                        <p>The optimization of the van Genuchten equation parameters is essential to ensure the accuracy and reproducibility of the estimates. Through an optimization algorithm, the parameters θr, α, and m are adjusted, and the model's quality is evaluated through error metrics.</p>
                        <p><strong>ERRORMAX</strong> is given by the formula:</p>
                        <p><strong>ERRORMAX = max |θi,estimated - θi,measured| for i = 1, 2, 3,...</strong></p>
                        <p>Where θi,estimated is the estimated retention value for the i-th suction point and θi,measured is the experimentally measured retention value.</p>
                        <p>The RMSE (Root Mean Squared Error), representing the square root of the mean squared error of the retention data estimates in the suction range considered by the SPSCS (30 cm - 18000 cm), is given by:</p>
                        <p><strong>RMSE30-18000 = √(1/n-3 Σ (θi,estimated - θi,measured)^2)</strong></p>
                        <p>Where n is the total number of data points used for the calculation in the suction range of 30 cm to 18000 cm.</p>
                        <p>With the validated parameters, the developed code generates the air availability curves used in the SPSCS classification. The microspace curves (A60) are calculated by <strong>A60 = 1 - (1 + (α⋅60)^n)^-m</strong>, the macrospace curves (W15000) by <strong>W15000 = (1 + (α⋅15000)^n)^-m</strong>, and the mesospace curves (W60 - W15000) by <strong>W60 - W15000 = (1 - A60) - W15000</strong>. The code then quantifies the percentage of each curve for each soil, allowing the classification of soils into nine orders, identified from A to I.</p>
                        <p>The suborder of each soil is determined based on the range of the air availability curve's scale factor, obtained from the difference between the water content at the saturation point (θs) and the residual water content (θr). The combination of orders and suborders classifies each soil into one of 36 possible families. The classified soils are then plotted on a ternary diagram, where the sides represent the percentages of microspace, mesospace, and macrospace. The code also generates specific ternary diagrams for each suborder, excluding other suborders.</p>
                        <p>Additionally, the system generates Excel files containing all the classified data, along with the water retention curves for each sample. These files facilitate data storage and analysis, allowing users to make comparisons between different soil samples.</p>
                        <p style="text-align: center;">
                            <img src="assets/instructions.png" alt="Instructions for using the menu" style="max-width: 16%; height: auto;">
                        </p>
                        <h3>In the Menu</h3>
                        <div style="display: flex; justify-content: flex-start; align-items: flex-start;">
                            <ul style="flex: 1; padding-right: 20px;">
                                <li style="margin-bottom: 20px;">After accessing the website, the user can choose between the options “Fill One Unique Soil” or “Import XLSX” located on the left side of the screen, as shown in the image.</li>
                                <li style="margin-bottom: 20px;">In the “Fill One Unique Soil” option, the user manually enters the suction values and volumetric water content for a single soil sample.</li>
                                <li style="margin-bottom: 20px;">In the “Import XLSX” option, the user uploads a .xlsx file with their database of suction values and volumetric water content corresponding to multiple soil samples. User input data is not stored in the system.</li>
                            </ul>
                            <img src="assets/Menu.png" alt="Menu Image" style="max-width: 40%; height: auto; margin-left: 20px;">
                        </div>
                        <h3>In the “Fill One Unique Soil” Option</h3>
                        <p style="text-align: center;">
                            <img src="assets/Fillunique1.png" alt="Instructions for using the menu" style="max-width: 90%; height: auto;">
                        </p>
                        <ul>
                            <li>In the “Fill One Unique Soil” option, a window called “Fill Form” appears, consisting of fields for entering the soil code and the number of retention points. The user must enter at least four retention points, with no upper limit on the number of points.</li>
                            <li>Next to the “Fill Form” window is the “Important Notes” window, which contains information about the minimum number of suction points that the user must enter and the criteria to be followed for these four points, described as follows: one point with suction equal to 0 cm, or very close to 0 (0&lt;h&gt;1); one suction point in the range of 30 to 80 cm; one suction point in the range of 250 to 500 cm; one suction point in the range of 9000 to 18000 cm. All suction values must be positive.</li>
                            <p>
                               <img src="assets/Fillunique2.png" alt="Instructions for using the menu" style="max-width: 45%; height: auto;">
                            </p>
                            <li>After entering the number of retention points, the “Fill Form” window expands to accommodate the columns for entering the suction data in the “Suction” column and their respective volumetric water content data in the “Volumetric water content” column. After entering all the data, the user must press the “Run” button to start processing.</li>
                        </ul>
                        <h3>After Processing Ends:</h3>
                        <ul>
                            <li>In the “Fill Form” window, the buttons “Download your Classified Soil” and “Download your Plots” will appear, which, when pressed, will result in downloading the Excel file “YourSoilClassified.xlsx” and the file “ternary_plots.zip,” respectively. The “YourSoilClassified” file will contain the classification of the entered soil following the SPSCS method, as well as other data such as the optimized van Genuchten parameters, percentages of meso, macro, and microspaces, etc. The “ternary_plots” file will contain the image of the soil plotted on the SPSCS ternary triangle, indicating its degree of adherence, order, and suborder, and the image of the retention curve generated by the optimization of the VG parameters.</li>
                            <li>Below the “Fill Form” window, the “Plots” window will appear, where the ternary triangle and retention curve images will be displayed.</li>
                        </ul>
                        <p>
                            <img src="assets/Fillunique3.png" alt="Instructions for using the menu" style="max-width: 60%; height: auto;">
                        </p>
                        <p style="text-align: center;">
                            <img src="assets/Fillunique4.png" alt="Instructions for using the menu" style="max-width: 80%; height: auto;">
                        </p>
                        <h3>In the “Import XLSX” Menu</h3>
                        <p style="text-align: center;">
                            <img src="assets/Import1.png" alt="Instructions for using the menu" style="max-width: 90%; height: auto;">
                        </p>
                        <ul>
                            <li>When opting to perform the calculation by importing an .xlsx file, the user will find the fields “Import Data” and “Instructions.”</li>
                            <li>The “Instructions” field contains examples of the two possible .xlsx file formats that can be uploaded into the program, one with the data filled in columns and the other in rows. The user can download either or both formats by pressing the “Download Column Template” and “Download Row Template</li>
                            <li>Each soil sample must contain at least 4 suction points, as described earlier: one point with suction equal to 0 cm, or very close to 0 (0&lt;h&gt;1); one suction point in the range of 30 to 80 cm; one suction point in the range of 250 to 500 cm; one suction point in the range of 9000 to 18000 cm. All suction values must be positive.</li>
                        </ul>
                        <p>In the “Import Data” field, after filling out and saving the chosen example file (which can be saved under any desired name), the user must select the “Choose file” option to access a directory to select the saved and filled file, and then click the “Run” button to start processing.</p>
                        <h3>After Processing Ends:</h3>
                        <ul>
                            <li>In the “Import Data” window, the buttons “Download your Classified Soil” and “Download your Plots” will appear, which, when pressed, will result in downloading the Excel file “YourSoilClassified.xlsx” and the file “ternary_plots.zip,” respectively. The “YourSoilClassified” file will contain the classification of the entered soils following the SPSCS method, as well as other data such as the optimized van Genuchten parameters, percentages of meso, macro, and microspaces, etc. The “ternary_plots” file will contain the images of the soils plotted on the SPSCS ternary triangle, indicating their degree of adherence, images of the ternary triangle for each suborder containing only the soils from the respective suborder, and a PDF file containing the retention curves for each soil generated by optimizing the VG parameters.</li>
                            <p>
                                <img src="assets/Import2.png" alt="Instructions for using the menu" style="max-width: 60%; height: auto;">
                            </p>
                            <li>Below the “Import Data” window, the “Plots” window will appear, where the images of the ternary triangles and retention curves will be displayed.</li>
                        </ul>
                        <p style="text-align: center;">
                            <img src="assets/Import3.png" alt="Instructions for using the menu" style="max-width: 60%; height: auto;">
                        </p>
                        <p style="text-align: center;">
                            <img src="assets/Import4.png" alt="Instructions for using the menu" style="max-width: 60%; height: auto;">
                        </p>
                        <p style="text-align: center;">
                            <img src="assets/Import5.png" alt="Instructions for using the menu" style="max-width: 60%; height: auto;">
                        </p>
                        <p style="text-align: center;">
                            <img src="assets/Import6.png" alt="Instructions for using the menu" style="max-width: 60%; height: auto;">
                        </p>
                    </div>
                `
            };
    

    content.innerHTML = titles[page];

    // Remove the 'active' class from all sidebar buttons
    let buttons = document.querySelectorAll('.sidebar button');
    buttons.forEach(button => button.classList.remove('active'));

    // Add the 'active' class to the clicked button
    let activeButton = document.querySelector(`.sidebar button[onclick="showPage('${page}')"]`);
    if (activeButton) {
        activeButton.classList.add('active');
    }
}

let isProcessing = false;


function generateInputRows() {
    let numPoints = parseInt(document.getElementById('num-points').value);

    // Validate the number of points
    if (numPoints < 4) {
        alert('Please enter a minimum of 4 points.'); // Alert if less than 4
        return; // Exit the function without generating input rows
    }

    let inputRows = document.getElementById('input-rows');
    inputRows.innerHTML = '';

    // Títulos únicos acima das colunas
    let row = document.createElement('div');
    row.style.display = 'flex'; // Usar flexbox para alinhar os títulos
    row.style.marginBottom = '10px'; // Espaçamento entre títulos e linhas

    let hLabel = document.createElement('label');
    hLabel.textContent = 'Suction (h, cm)*';
    hLabel.style.marginRight = '30px'; // Adicionar espaço à direita

    let thetaLabel = document.createElement('label');
    thetaLabel.textContent = 'Volumetric water content (cm³/cm³)';
    thetaLabel.style.marginLeft = '80px'; // Adicionar margem à esquerda de 140 pixels
    thetaLabel.style.marginTop = '0px'; // Manter na mesma linha

    row.appendChild(hLabel);
    row.appendChild(thetaLabel);
    inputRows.appendChild(row);

    for (let i = 0; i < numPoints; i++) {
        let inputRow = document.createElement('div');
        inputRow.className = 'input-row';
        inputRow.style.display = 'block'; // Alterado para block para evitar a centralização
        inputRow.style.marginTop = '5px'; // Aumentar o espaçamento entre as linhas

        let hInput = document.createElement('input');
        hInput.type = 'number';
        hInput.placeholder = 'h(cm)';
        hInput.className = 'h-input';
        hInput.style.width = '30%'; // Tamanho reduzido da caixa de entrada
        hInput.style.marginRight = '30px'; // Definir a margem entre as caixas de entrada para 30 pixels

        let thetaInput = document.createElement('input');
        thetaInput.type = 'number';
        thetaInput.placeholder = 'theta(cm³/cm³)';
        thetaInput.className = 'theta-input';
        thetaInput.style.width = '30%';
        thetaInput.style.marginLeft = '0px'; // Ajustar a margem à esquerda para aproximar da caixa h

        inputRow.appendChild(hInput);
        inputRow.appendChild(thetaInput);
        inputRows.appendChild(inputRow);
    }
}

function createExcel() {
    let soilCode = document.getElementById('soil-code').value.trim();
    let numPoints = parseInt(document.getElementById('num-points').value);
    let hInputs = document.querySelectorAll('.h-input');
    let thetaInputs = document.querySelectorAll('.theta-input');

    let errorMessages = []; // Array to collect error messages

    if (!soilCode) {
        errorMessages.push('Please enter the Soil Code.');
    }

    if (isNaN(numPoints) || numPoints <= 0) {
        errorMessages.push('Please enter a valid Number of Points.');
    }

    if (numPoints < 4) {
        errorMessages.push('You must enter at least 4 points.');
    }

    if (hInputs.length !== numPoints || thetaInputs.length !== numPoints) {
        errorMessages.push('The number of h(cm) and theta inputs must equal the Number of Points.');
    }

    let hValues = [];
    let thetaValues = [];

    for (let i = 0; i < numPoints; i++) {
        let h = hInputs[i].value;
        let theta = thetaInputs[i].value;

        if (h === '' || theta === '') {
            errorMessages.push('Please fill in all fields for h(cm) and theta.');
            continue; // Skip to the next iteration
        }

        hValues.push(parseFloat(h));
        thetaValues.push(parseFloat(theta));
    }

    // Additional validations
    if (!hValues.includes(0)) {
        errorMessages.push('There must be at least one point where h equals 0.');
    }

    if (!hValues.some(h => h >= 30 && h <= 80)) {
        errorMessages.push('There must be at least one point with h between 30 and 80 cm.');
    }

    if (!hValues.some(h => h >= 250 && h <= 500)) {
        errorMessages.push('There must be at least one point with h between 250 and 500 cm.');
    }

    if (!hValues.some(h => h >= 9000 && h <= 15000)) {
        errorMessages.push('There must be at least one point with h between 9000 and 15000 cm.');
    }

    // If there are any error messages, alert the user and return
    if (errorMessages.length > 0) {
        alert(errorMessages.join('\n'));
        return;
    }

    // Show loading indicator
    document.getElementById('loading-indicator').style.display = 'block';
    document.getElementById('stopButton').style.display = 'inline'; // Show Stop button
    isProcessing = true; // Set processing flag to true

    // Send data to server to create the Excel file
    fetch('/create_excel', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            soil_code: soilCode,
            num_points: numPoints,
            h_values: hValues,
            theta_values: thetaValues
        })
    })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            document.getElementById('loading-indicator').style.display = 'none';
            document.getElementById('stopButton').style.display = 'none'; // Hide Stop button
            isProcessing = false; // Reset processing flag

            if (data.success) {
                alert('Excel file created successfully!');

                // Show download buttons after creating the Excel file
                document.getElementById('file1').href = '/download_xlsx'; // URL for YourSoilClassified
                document.getElementById('file2').href = '/download_zip'; // URL for Ternary Plots
                document.getElementById('file1').style.display = 'inline';
                document.getElementById('file2').style.display = 'inline';
                document.getElementById('form-download-buttons').style.display = 'block'; // Ensure download buttons are visible

                // Automatically show the plots container after processing using the new function
                visualizarPlot(); // Alterado para a nova função visualizarPlot()

            } else {
                alert('Error creating the Excel file.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing the request.');
            // Hide loading indicator in case of error
            document.getElementById('loading-indicator').style.display = 'none';
            document.getElementById('stopButton').style.display = 'none'; // Hide Stop button
            isProcessing = false; // Reset processing flag
        });
}

function stopProcessing() {
    // Implement stopping logic here
    if (isProcessing) {
        alert('Processing stopped.');
        isProcessing = false;
        document.getElementById('loading-indicator').style.display = 'none'; // Hide loading indicator
        document.getElementById('stopButton').style.display = 'none'; // Hide Stop button
        // Optionally, reset fields or perform any necessary cleanup
    }
}

function uploadFile() {
    let fileInput = document.getElementById('fileInput');
    let file = fileInput.files[0];
    if (file) {
        let formData = new FormData();
        formData.append('file', file);

        // Show loading indicator
        document.getElementById('loading-indicator-upload').style.display = 'block';
        document.getElementById('stopButtonUpload').style.display = 'inline'; // Show Stop button
        isProcessing = true; // Set processing flag to true

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.getElementById('loading-indicator-upload').style.display = 'none';
                document.getElementById('stopButtonUpload').style.display = 'none'; // Hide Stop button
                isProcessing = false; // Reset processing flag

                if (data.success) {
                    alert('File uploaded successfully!');

                    // Show download links for files
                    document.getElementById('file1').href = '/download_xlsx'; // Link for YourSoilClassified
                    document.getElementById('file2').href = '/download_zip'; // Link for Ternary Plots
                    document.getElementById('file1').style.display = 'inline';
                    document.getElementById('file2').style.display = 'inline';

                    // Automatically show the plots container after uploading
                    visualizarPlots();

                } else {
                    alert('Error uploading the file.');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Make sure that your file follows one of the formats and that at least two of the soils have points in the expecified ranges (See About Us).');
                // Hide loading indicator in case of error
                document.getElementById('loading-indicator-upload').style.display = 'none';
                document.getElementById('stopButtonUpload').style.display = 'none'; // Hide Stop button
                isProcessing = false; // Reset processing flag
            });
    } else {
        alert('Please select a file to upload.');
    }
}

function stopUploadProcessing() {
    // Implement stopping logic here for upload
    if (isProcessing) {
        alert('Upload processing stopped.');
        isProcessing = false;
        document.getElementById('loading-indicator-upload').style.display = 'none'; // Hide loading indicator
        document.getElementById('stopButtonUpload').style.display = 'none'; // Hide Stop button
        // Optionally, reset fields or perform any necessary cleanup
    }
}

let isPlotsVisible = false;

// Função para mostrar ou esconder o contêiner de plots
function visualizarPlots() {
    let content = document.getElementById('content');

    if (isPlotsVisible) {
        // Se os plots estão visíveis, remove o contêiner
        const plotsContainer = document.getElementById('plots-container');
        if (plotsContainer) {
            plotsContainer.remove();
        }
        isPlotsVisible = false;
    } else {
        // Se os plots não estão visíveis, cria o contêiner
        const plotsContainer = document.createElement('div');
        plotsContainer.id = 'plots-container';
        plotsContainer.className = 'formulario-container';

        plotsContainer.innerHTML = `
            <h2>Plots</h2>
            <div class="images-container" style="text-align: center;">
                <img src="ternary_plots/allsuborders_plot.jpg" alt="Ternary Plot" class="main-plot" style="max-width: 40%; height: auto; margin-bottom: 10px;">
                <div class="small-images" style="display: flex; justify-content: center; flex-wrap: wrap; margin-top: 10px;">
                    <div style="margin: 5px;">
                        <img src="ternary_plots/suborder_1_plot.jpg" alt="Order 1 Plot" style="max-width: 100%; height: auto;">
                    </div>
                    <div style="margin: 5px;">
                        <img src="ternary_plots/suborder_2_plot.jpg" alt="Order 2 Plot" style="max-width: 100%; height: auto;">
                    </div>
                    <div style="margin: 5px;">
                        <img src="ternary_plots/suborder_3_plot.jpg" alt="Order 3 Plot" style="max-width: 100%; height: auto;">
                    </div>
                    <div style="margin: 5px;">
                        <img src="ternary_plots/suborder_4_plot.jpg" alt="Order 4 Plot" style="max-width: 100%; height: auto;">
                    </div>
                </div>
                <!-- Adiciona o título para o PDF -->
                <h3 style="color: gray; margin-top: 20px;">Retention Curves</h3>
                <!-- Adiciona o PDF embutido abaixo das imagens -->
                <embed src="ternary_plots/retentioncurveplots.pdf" type="application/pdf" style="width: 100%; height: 450px; margin-top: 20px;">
            </div>
        `;

        // Adiciona o contêiner de plots ao conteúdo
        content.appendChild(plotsContainer);
        isPlotsVisible = true;
    }
}

function visualizarPlot() {
    let content = document.getElementById('content');

    if (isPlotsVisible) {
        // Se os plots estão visíveis, remove o contêiner
        const plotsContainer = document.getElementById('plots-container');
        if (plotsContainer) {
            plotsContainer.remove();
        }
        isPlotsVisible = false;
    } else {
        // Se os plots não estão visíveis, cria o contêiner
        const plotsContainer = document.createElement('div');
        plotsContainer.id = 'plots-container';
        plotsContainer.className = 'formulario-container';

        // Cria o contêiner para o gráfico Ternary
        const ternaryPlotContainer = document.createElement('div');
        ternaryPlotContainer.style.width = '50%';
        ternaryPlotContainer.innerHTML = `
            <h2>Plots</h2>
            <img src="ternary_plots/suborder_plot_1soil.jpg" alt="Order" style="max-width: 100%; height: auto; margin: 0;">
        `;

        // Cria o contêiner para o gráfico Retention Curve
        const retentionCurveContainer = document.createElement('div');
        retentionCurveContainer.style.width = '50%';
        retentionCurveContainer.style.marginTop = '20px'; // Adiciona margem superior de 10 pixels
        retentionCurveContainer.innerHTML = `
            <img src="ternary_plots/retention_curve_code_1.jpg" alt="Theta vs Log(h)" style="max-width: 100%; height: auto; margin: 0;">
        `;

        // Adiciona ambos os contêineres ao contêiner principal
        plotsContainer.style.display = 'flex'; // Flexbox para alinhamento em linha
        plotsContainer.appendChild(ternaryPlotContainer);
        plotsContainer.appendChild(retentionCurveContainer);

        // Adiciona o contêiner de plots ao conteúdo
        content.appendChild(plotsContainer);
        isPlotsVisible = true;
    }
}
// Chama showPage ao carregar a página
window.onload = function () {
    showPage('preencher-solo');
};
