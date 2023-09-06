<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Sales Prediction</title>

    <!-- Bootstrap CSS -->
    <link
      rel="stylesheet"
      href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css"
      integrity="sha384-9aIt2nRpC12Uk9gS9baDl411NQApFmC26EwAOH8WgZl5MYYxFfc+NcPb1dKGj7Sk"
      crossorigin="anonymous"
    />

    <!-- Custom CSS -->
    <link rel="stylesheet" href="static/css/styles.css" />
  </head>
  <body>
    <!-- Navigation Bar -->
    <nav class="navbar navbar-inverse navbar-fixed-top">
      <div class="container-fluid">
        <div class="navbar-header">
          <a class="navbar-brand" href="/">Back to home page</a>
        </div>
      </div>
    </nav>

    <!-- Navigation Bar -->
    <nav class="navbar navbar-inverse navbar-fixed-top">
      <div class="container-fluid">
        <div class="navbar-header">
          <h3 class="navbar-brand">Big Mart Sales Price Predictor Machine</h3>
        </div>
      </div>
    </nav>

    <!-- Navigation Bar -->
    <nav class="navbar navbar-inverse navbar-fixed-top">
      <div class="container-fluid">
        <div class="navbar-header">
          <h4 class="navbar-brand">
            Input your features and get your sales prediction
          </h4>
        </div>
      </div>
    </nav>
    <!-- Form Container -->
    <div class="container">
      <form action="{{ url_for('predict_datapoint')}}" method="post">
        <div class="col-sm-6">
          <div class="card">
            <div class="card-body">
              <h6 class="card-title">
                What is the Item Weight of the Product?
              </h6>
              <!-- Item_Weight -->
              <input
                type="number"
                name="Item_Weight"
                min="4.56"
                max="21.35"
                step="1"
                placeholder="Enter a number"
                class="form-control"
              />

              <h6 class="card-title">Is the Product Low Fat or Regular?</h6>
              <!-- Item_Fat_Content Dropdown -->
              <div class="form-group">
                <select
                  name="Item_Fat_Content"
                  id="Item_Fat_Content"
                  class="form-control"
                >
                  <option value="Low Fat">Low Fat</option>
                  <option value="Regular">Regular</option>
                </select>
              </div>

              <h6 class="card-title">Category of the Product</h6>
              <!-- Item_Type Dropdown -->
              <div class="form-group">
                <select name="Item_Type" id="Item_Type" class="form-control">
                  <option value="Dairy">Dairy</option>
                  <option value="Soft Drinks">Soft Drinks</option>
                  <option value="Meat">Meat</option>
                  <option value="Fruits and Vegetables">
                    Fruits and Vegetables
                  </option>
                  <option value="Household">Household</option>
                  <option value="Baking Goods">Baking Goods</option>
                  <option value="Frozen Foods">Frozen Foods</option>
                  <option value="Breakfast">Breakfast</option>
                  <option value="Health and Hygiene">Health and Hygiene</option>
                  <option value="Hard Drinks">Hard Drinks</option>
                  <option value="Canned">Canned</option>
                  <option value="Breads">Breads</option>
                  <option value="Starchy Foods">Starchy Foods</option>
                  <option value="Others">Others</option>
                  <option value="Seafood">Seafood</option>
                </select>
              </div>

              <h6 class="card-title">
                What is the Retail Price of the Product?
              </h6>
              <!-- Item_MRP -->
              <input
                type="number"
                name="Item_MRP"
                min="31.49"
                max="266.89"
                step="1"
                placeholder="Enter a number"
                class="form-control"
              />

              <h6 class="card-title">Store Size (Ground Area Covered)</h6>
              <!-- Outlet_Size Dropdown -->
              <div class="form-group">
                <select
                  name="Outlet_Size"
                  id="Outlet_Size"
                  class="form-control"
                >
                  <option value="Medium">Medium</option>
                  <option value="High">High</option>
                  <option value="Small">Small</option>
                </select>
              </div>

              <h6 class="card-title">Outlet Location Type</h6>
              <!-- Outlet_Location_Type Dropdown -->
              <div class="form-group">
                <select
                  name="Outlet_Location_Type"
                  id="Outlet_Location_Type"
                  class="form-control"
                >
                  <option value="Tier 1">Tier 1</option>
                  <option value="Tier 2">Tier 2</option>
                  <option value="Tier 3">Tier 3</option>
                </select>
              </div>

              <h6 class="card-title">Outlet Type</h6>
              <!-- Outlet_Type Dropdown -->
              <div class="form-group">
                <select
                  name="Outlet_Type"
                  id="Outlet_Type"
                  class="form-control"
                >
                  <option value="Supermarket Type1">Supermarket Type1</option>
                  <option value="Supermarket Type2">Supermarket Type2</option>
                  <option value="Grocery Store">Grocery Store</option>
                  <option value="Supermarket Type3">Supermarket Type3</option>
                </select>
              </div>

              <h6 class="card-title">Age of the Store</h6>
              <!-- Outlet_Age -->
              <input
                type="number"
                name="Outlet_Age"
                min="14"
                max="36"
                step="1"
                placeholder="Enter a number"
                class="form-control"
              />
            </div>
          </div>
        </div>

        <!-- Submit Button -->
        <br /><br /><br />
        <input
          type="Get Prediction"
          value="Get Prediction"
          class="btn btn-secondary"
        />
      </form>

      <!-- Results Display -->
      <br /><br />
      <h3>{{ results }}</h3>

      <!-- Footer -->
      <br /><br />
      <p>Ayodele Ayodeji</p>
    </div>

    <!-- Bootstrap and JavaScript -->
    <script
      src="https://code.jquery.com/jquery-3.5.1.slim.min.js"
      integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj"
      crossorigin="anonymous"
    ></script>
    <script
      src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js"
      integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo"
      crossorigin="anonymous"
    ></script>
    <script
      src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js"
      integrity="sha384-OgVRvuATP1z7JjHLkuOU7Xw704+h835Lr+6QL9UvYjZE3Ipu6Tp75j7Bh/kR0JKI"
      crossorigin="anonymous"
    ></script>
  </body>
</html>