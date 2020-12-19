var svg = d3.select("svg"),
  margin = 20,
  diameter = +svg.attr("width"),
  g = svg.append("g").attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")");

var color = d3.scaleLinear()
  .domain([-1, 5])
  .range(["hsl(152,80%,80%)", "hsl(228,30%,40%)"])
  .interpolate(d3.interpolateHcl);

const colorScale = d3.scaleOrdinal()
  .domain(["one", "two", "three", "four", "five"])
  .range(d3.schemeCategory10);

var pack = d3.pack()
  .size([diameter - margin, diameter - margin])
  .padding(2);

d3.json("data/processed/burritos.json").then(function (root) {
  root = d3.hierarchy(root)
    .sum(function (d) { return d.overall; })
    .sort(function (a, b) { return b.overall - a.overall; });

  var focus = root,
    nodes = pack(root).descendants(),
    view;

  var circle = g.selectAll("circle")
    .data(nodes)
    .enter()
    .append("circle")
    .attr("class", function (d) { return d.parent ? d.children ? "node" : "node node--leaf" : "node node--root"; })
    .style("fill", function (d) {
      return d.children ? color(d.depth) : colorScale(getReviewBin(d.data.overall));
    })
    .on("click", function (d) { if (focus !== d) zoom(d), d3.event.stopPropagation(); })
    .on('mouseover', function (d, i) {
      updateDetailInfo(d.data, d.depth)
    })
    .on('mouseout', function (d, i) {

    });

  var text = g.selectAll("text")
    .data(nodes)
    .enter().append("text")
    .attr("class", "label")
    .style("fill-opacity", function (d) { return d.parent === root ? 1 : 0; })
    .style("display", function (d) { return d.parent === root ? "inline" : "none"; })
    .html(function (d) { return getNodeText(d.children, d.data); });

  var node = g.selectAll("circle,text");

  svg
    .on("click", function () { zoom(root); });

  zoomTo([root.x, root.y, root.r * 2 + margin]);

  function zoom(d) {
    var focus0 = focus; focus = d;

    var transition = d3.transition()
      .duration(d3.event.altKey ? 7500 : 750)
      .tween("zoom", function (d) {
        var i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2 + margin]);
        return function (t) { zoomTo(i(t)); };
      });

    transition.selectAll("text")
      .filter(function (d) { return d.parent === focus || this.style.display === "inline"; })
      .style("fill-opacity", function (d) { return d.parent === focus ? 1 : 0; })
      .on("start", function (d) { if (d.parent === focus) this.style.display = "inline"; })
      .on("end", function (d) { if (d.parent !== focus) this.style.display = "none"; });
  }

  function zoomTo(v) {
    var k = diameter / v[2]; view = v;
    node.attr("transform", function (d) { return "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")"; });
    circle.attr("r", function (d) { return d.r * k; });
  }
});

function getNodeText(hasChildren, data) {
  if (hasChildren) {
    return data.name;
  } else {
    return data.name + ' Burrito'
  }
}

function getReviewBin(value) {
  if (value >= 4.0) {
    return "five"
  } else if (value >= 3.0) {
    return "four"
  } else if (value >= 2.0) {
    return "three"
  } else if (value >= 1.0) {
    return "two"
  } else {
    return "one"
  }
}

function updateDetailInfo(data, depth) {
  if (depth == 1) {
    showNeighbourhoodInfo(data)
  } else if (depth == 2) {
    showLocationInfo(data)
  } else {
    showBurritoInfo(data)
  }
}

function showNeighbourhoodInfo(data) {
  document.getElementById('neighbourhood').style.visibility = 'visible'
  document.getElementById('location').style.visibility = 'hidden'
  document.getElementById('burrito').style.visibility = 'hidden'

  document.getElementById('n_name').innerHTML = data.name
  document.getElementById('n_location_count').innerHTML = data.children.length
  document.getElementById('n_burrito_count').innerHTML = data.burrito_count
}

function showLocationInfo(data) {
  document.getElementById('neighbourhood').style.visibility = 'hidden'
  document.getElementById('location').style.visibility = 'visible'
  document.getElementById('burrito').style.visibility = 'hidden'

  document.getElementById('l_name').innerHTML = data.name
  document.getElementById('l_google_rating').innerHTML = data.rating

  const starPercentage = (data.rating / 5) * 100;

  const starPercentageRounded = `${(Math.round(starPercentage / 10) * 10)}%`;
  document.querySelector(`.location .stars-inner`).style.width = starPercentageRounded;
}

function showBurritoInfo(data) {
  document.getElementById('neighbourhood').style.visibility = 'hidden'
  document.getElementById('location').style.visibility = 'hidden'
  document.getElementById('burrito').style.visibility = 'visible'

  document.getElementById('b_name').innerHTML = data.name + ' Burrito'
  document.getElementById('b_overall').innerHTML = '' + data.overall

  const starPercentage = (data.overall / 5) * 100;

  const starPercentageRounded = `${(Math.round(starPercentage / 10) * 10)}%`;
  document.querySelector(`.burrito .stars-inner`).style.width = starPercentageRounded;

  for (const rating in data.ingredients) {
    const starPercentage = (data.ingredients[rating] / 5.0) * 100;
    const starPercentageRounded = `${(Math.round(starPercentage / 10) * 10)}%`;
    document.querySelector(`.${rating} .stars-inner`).style.width = starPercentageRounded;
  }

  document.getElementById('b_cost').innerHTML = getCost(data.Cost)
}


function getCost(data) {
  var cost = data.Cost;

  if (cost > 20) {
    return "<span>&#36; &#36;  &#36;</span>"
  } else if (cost > 10) {
    return "<span>&#36; &#36;</span>"
  } else {
    return "<span>&#36;</span>"
  }
}

var legend = d3.legendColor()
  .scale(colorScale)
  .shapePadding(12)
  .title("Burrito Ratings Legend:")
  .titleWidth(200)
  .shapeWidth(25)
  .shapeHeight(25)
  .labelOffset(25)
  .cells(5);

d3.select("#ratings-legend").append("g")
  .attr("transform", "translate(200,10)")
  .call(legend);