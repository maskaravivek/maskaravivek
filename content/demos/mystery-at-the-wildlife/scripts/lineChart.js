export {drawLineChart}
import { drawViolinPlot } from './box.js'
let margin = {top: 10, right: 30, bottom: 50, left: 60}
let width = 460 - margin.left - margin.right
let height = 400 - margin.top - margin.bottom
const timeConv = d3.timeParse("%Y-%m-%d");
let data
var sensorData
let hidden = {
    'AGOC 3A' : false,
    'Appluimonia' : false,
    'Chlorodinine' : false,
    'Methylosmolene' : false,
}
var sensor_no

let svg = d3.select("#graph-layout")
    .append("div")
    .append("svg")
    .attr("class","line-svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform","translate(" + margin.left + "," + margin.top + ")");
    

var tooltip = d3.select("#linechart")
                .append("div")
                .attr("class", "tooltip")
                .style("opacity", 0);



document.addEventListener("DOMContentLoaded", async function () {
    let values = await Promise.all([
        d3.csv("data/processed_sensor_data_line_chart.csv"),
        d3.csv('data/sensor_data.csv')
    ]);
    data = values[0]
    sensorData = values[1]
    var month = document.getElementById("months-linechart").value;
    sensor_no = "1"
    drawLineChart(month,sensor_no)

});

function clear() {
    d3.select(".line-svg").selectAll("*").remove();
    d3.select("#violinSvg").remove();
    svg = d3
      .select(".line-svg")
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);
  }

function drawLineChart(monthName,sensor) {
    clear()
    sensor_no = sensor
    
    hidden['Methylosmolene'] = false
    hidden['Chlorodinine'] = false
    hidden['Appluimonia'] = false
    hidden['AGOC 3A'] = false

    var methylosmoleneText = document.getElementById('text-METHYLOSMOLENE');
    methylosmoleneText.style.opacity = 1;

    var chlorodinineText = document.getElementById('text-CHLORODININE');
    chlorodinineText.style.opacity = 1;

    var appluimoniaText = document.getElementById('text-APPLUIMONIA');
    appluimoniaText.style.opacity = 1;

    var agocText = document.getElementById('text-AGOC3A');
    agocText.style.opacity = 1;


    var months = [ 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december' ];
    var month = (months.indexOf(monthName));      

    let xScale = d3.scaleTime()
    .range([ 0, width ])
    .domain(d3.extent(data, function(d) { 
        var xDate =  timeConv(d.Date)
        if(xDate.getMonth()===month && d.Monitor===sensor) 
            return xDate;
    }))

    let yScale = d3.scaleLog()
    .range([ height, 0 ])
    .domain([d3.min(data, function (d) {
            return +d.Reading
    }), d3.max(data, function(d) { 
            return +d.Reading; 
    })])

    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(xScale).ticks(3));

    svg.append("g")
        .call(d3.axisLeft(yScale).ticks(11,".1n"));


    var lineGen = d3.line()
        .x(function(d) { 
            return xScale(timeConv(d.Date)); 
        })
        .y(function(d) { 
            return yScale(+d.Reading); 
        })
        .curve(d3.curveBasis);
    
    let AGOC_3A_data = []
    let Appluimonia_data = []
    let Chlorodinine_data = []
    let Methylosmolene_data = []
    // console.log(data)
    data.forEach(element => {
        var mon = timeConv(element['Date']).getMonth()
        if (element['Chemical'] == 'AGOC-3A' && month===mon && element['Monitor']===sensor) {
            AGOC_3A_data.push(element)
        } else if (element['Chemical'] == 'Appluimonia' && month===mon && element['Monitor']===sensor) {
            Appluimonia_data.push(element)
        } else if (element['Chemical'] == 'Chlorodinine' && month===mon && element['Monitor']===sensor) {
            Chlorodinine_data.push(element)
        } else if (element['Chemical'] == 'Methylosmolene' && month===mon && element['Monitor']===sensor) {
            Methylosmolene_data.push(element)
        }
    })

    svg.append('svg:path')
        .attr('d', lineGen(AGOC_3A_data))
        .attr('stroke', 'green')
        .attr('stroke-width', 3)
        .attr('fill', 'none')
        .attr("id", "line-AGOC3A")
        .on('mouseover', function (d, i, j) {
            const monthNames = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ];

            var coords = d3.mouse(this);
            var newData= {
                x:  xScale.invert(coords[0]),  
                y: yScale.invert(coords[1])
            };
            d3.select("#line-AGOC3A").transition()
            .duration('150')
            .attr('cursor', 'pointer')
            tooltip.transition()
            .duration(50)
            .style("opacity", 1);
            tooltip.html("Sensor: "+ sensor +"<br>Chemical: " +"Agoc 3A" + "<br>Date: "+monthNames[newData.x.getMonth()]+" "+ newData.x.getDate()+"<br>Reading: "+newData.y.toFixed(4))
            .style("left", (d3.event.pageX) + "px")
            .style("top", (d3.event.pageY) + "px");
    
        })
        .on('mouseout', function (d, i) {
            tooltip.transition()
				.duration(50)
                .style("opacity", 0);
        })
        .on("click", function(d) { 
            drawViolinPlot(sensorData,'AGOC-3A',sensor_no)
        });

    svg.append('svg:path')
        .attr('d', lineGen(Appluimonia_data))
        .attr('stroke', 'blue')
        .attr('stroke-width', 3)
        .attr('fill', 'none')
        .attr("id", "line-APPLUIMONIA")
        .on('mouseover', function (d, i, j) {
            const monthNames = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ];

            var coords = d3.mouse(this);
            var newData= {
                x:  xScale.invert(coords[0]),  
                y: yScale.invert(coords[1])
            };
            d3.select("#line-APPLUIMONIA").transition()
            .duration('150')
            .attr('cursor', 'pointer')
            tooltip.transition()
            .duration(50)
            .style("opacity", 1);
            tooltip.html("Sensor: "+ sensor +"<br>Chemical: " +"Appluimonia" + "<br>Date: "+monthNames[newData.x.getMonth()] +" "+ newData.x.getDate()+"<br>Reading: "+newData.y.toFixed(4))
            .style("left", (d3.event.pageX) + "px")
            .style("top", (d3.event.pageY) + "px");
    
        })
        .on('mouseout', function (d, i) {
            tooltip.transition()
				.duration(50)
                .style("opacity", 0);
        })
        .on("click", function(d) { 
            drawViolinPlot(sensorData,'Appluimonia',sensor_no)
        });

    svg.append('svg:path')
        .attr('d', lineGen(Chlorodinine_data))
        .attr('stroke', 'red')
        .attr('stroke-width', 3)
        .attr('fill', 'none')
        .attr("id", "line-CHLORODININE")
        .on('mouseover', function (d, i, j) {
            const monthNames = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ];

            var coords = d3.mouse(this);
            var newData= {
                x:  xScale.invert(coords[0]),  
                y: yScale.invert(coords[1])
            };
            d3.select("#line-CHLORODININE").transition()
            .duration('150')
            .attr('cursor', 'pointer')
            tooltip.transition()
            .duration(50)
            .style("opacity", 1);
            tooltip.html("Sensor: "+ sensor +"<br>Chemical: " +"Chlorodinine" + "<br>Date: "+monthNames[newData.x.getMonth()] +" "+ newData.x.getDate()+"<br>Reading: "+newData.y.toFixed(4))
            .style("left", (d3.event.pageX) + "px")
            .style("top", (d3.event.pageY) + "px");
    
        })
        .on('mouseout', function (d, i) {
            tooltip.transition()
				.duration(50)
                .style("opacity", 0);
        })
        .on("click", function(d) { 
            drawViolinPlot(sensorData,'Chlorodinine',sensor_no)
        });

    svg.append('svg:path')
        .attr('d', lineGen(Methylosmolene_data))
        .attr('stroke', 'orange')
        .attr('stroke-width', 3)
        .attr('fill', 'none')
        .attr("id", "line-METHYLOSMOLENE")
        .on('mouseover', function (d, i, j) {
            const monthNames = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ];
            var coords = d3.mouse(this);
            var newData= {
                x:  xScale.invert(coords[0]),  
                y: yScale.invert(coords[1])
            };
            d3.select("#line-METHYLOSMOLENE").transition()
            .duration('150')
            .attr('cursor', 'pointer')
            tooltip.transition()
            .duration(50)
            .style("opacity", 1);
            tooltip.html("Sensor: "+ sensor +"<br>Chemical: " +"Methylosmolene" + "<br>Date: "+monthNames[newData.x.getMonth()] +" "+ newData.x.getDate()+"<br>Reading: "+newData.y.toFixed(4))
            .style("left", (d3.event.pageX) + "px")
            .style("top", (d3.event.pageY) + "px");
    
        })
        .on('mouseout', function (d, i) {
            tooltip.transition()
				.duration(50)
                .style("opacity", 0);
        })
        .on("click", function(d) { 
            drawViolinPlot(sensorData,'Methylosmolene',sensor_no)
        });


    svg.append("text")
        .attr("class", "x label")
        .attr("text-anchor", "end")
        .attr("x", width-190)
        .style("font-size", "14px") 
        .style("font-weight", "bold")
        .style("font-family", "sans-serif")  
        .attr("y", 370)
        .attr("dy", ".75em")
        .attr("transform", "rotate(0)")
        .text("Time");

    svg.append("text")
        .attr("class", "y label")
        .attr("text-anchor", "end")
        .attr("x", -120)
        .style("font-size", "14px") 
        .style("font-weight", "bold") 
        .style("font-family", "sans-serif") 
        .attr("y", -50)
        .attr("dy", ".75em")
        .attr("transform", "rotate(-90)")
        .text("Chemical Reading Value");


    svg.append("text")
        .attr("x", (width / 2)+50)             
        .attr("y", 4 )
        .attr("text-anchor", "middle")  
        .style("font-size", "16px") 
        .style("font-weight", "bold") 
        .style("font-family", "sans-serif") 
        .style("text-decoration", "underline")  
        .text("Avg. Reading per day vs Time (Sensor " +sensor_no +")");    
}


document.getElementById("text-AGOC3A").addEventListener('click',function (){
   var agocLine = document.getElementById('line-AGOC3A');
   var agocText = document.getElementById('text-AGOC3A');
   if (hidden['AGOC 3A']) {
       agocLine.style.visibility = 'visible';
       agocText.style.opacity = 1;
    } else {
        agocLine.style.visibility = 'hidden';
        agocText.style.opacity = 0.2;
    }
    hidden['AGOC 3A'] = !hidden['AGOC 3A']
}); 
document.getElementById("text-AGOC3A").addEventListener('mouseover',function (){
    var agocText = document.getElementById('text-AGOC3A');
    agocText.style.opacity = 0.4;
    agocText.style.cursor = 'pointer';
});

document.getElementById("text-AGOC3A").addEventListener('mouseout',function (){
    var agocText = document.getElementById('text-AGOC3A');
    if (!hidden['AGOC 3A']) 
        agocText.style.opacity = 1;
    else 
        agocText.style.opacity = 0.2
});


document.getElementById("text-APPLUIMONIA").addEventListener('click',function (){
    var appluimoniaLine = document.getElementById('line-APPLUIMONIA');
    var appluimoniaText = document.getElementById('text-APPLUIMONIA');
    if (hidden['Appluimonia']) {
        appluimoniaLine.style.visibility = 'visible';
        appluimoniaText.style.opacity = 1;
     } else {
        appluimoniaLine.style.visibility = 'hidden';
        appluimoniaText.style.opacity = 0.2;
     }
    hidden['Appluimonia'] = !hidden['Appluimonia']
}); 

document.getElementById("text-APPLUIMONIA").addEventListener('mouseover',function (){
    var appluimoniaText = document.getElementById('text-APPLUIMONIA');
    appluimoniaText.style.opacity = 0.4;
    appluimoniaText.style.cursor = 'pointer';
});

document.getElementById("text-APPLUIMONIA").addEventListener('mouseout',function (){
    var appluimoniaText = document.getElementById('text-APPLUIMONIA');
    if (!hidden['Appluimonia']) 
        appluimoniaText.style.opacity = 1;
    else 
        appluimoniaText.style.opacity = 0.2
});


document.getElementById("text-CHLORODININE").addEventListener('click',function (){
    var chlorodinineLine = document.getElementById('line-CHLORODININE');
    var chlorodinineText = document.getElementById('text-CHLORODININE');
    if (hidden['Chlorodinine']) {
        chlorodinineLine.style.visibility = 'visible';
        chlorodinineText.style.opacity = 1;
     } else {
        chlorodinineLine.style.visibility = 'hidden';
        chlorodinineText.style.opacity = 0.2;
     }
    hidden['Chlorodinine'] = !hidden['Chlorodinine']
}); 

document.getElementById("text-CHLORODININE").addEventListener('mouseover',function (){
    var chlorodinineText = document.getElementById('text-CHLORODININE');
    chlorodinineText.style.opacity = 0.4;
    chlorodinineText.style.cursor = 'pointer';
});

document.getElementById("text-CHLORODININE").addEventListener('mouseout',function (){
    var chlorodinineText = document.getElementById('text-CHLORODININE');
    if (!hidden['Chlorodinine']) 
        chlorodinineText.style.opacity = 1;
    else 
        chlorodinineText.style.opacity = 0.2
});


document.getElementById("text-METHYLOSMOLENE").addEventListener('click',function (){
    var methylosmoleneLine = document.getElementById('line-METHYLOSMOLENE');
    var methylosmoleneText = document.getElementById('text-METHYLOSMOLENE');
    if (hidden['Methylosmolene']) {
        methylosmoleneLine.style.visibility = 'visible';
        methylosmoleneText.style.opacity = 1;
     } else {
        methylosmoleneLine.style.visibility = 'hidden';
        methylosmoleneText.style.opacity = 0.2;
     }
    hidden['Methylosmolene'] = !hidden['Methylosmolene']
}); 

document.getElementById("text-METHYLOSMOLENE").addEventListener('mouseover',function (){
    var methylosmoleneText = document.getElementById('text-METHYLOSMOLENE');
    methylosmoleneText.style.opacity = 0.4;
    methylosmoleneText.style.cursor = 'pointer';
});

document.getElementById("text-METHYLOSMOLENE").addEventListener('mouseout',function (){
    var methylosmoleneText = document.getElementById('text-METHYLOSMOLENE');
    if (!hidden['Methylosmolene']) 
        methylosmoleneText.style.opacity = 1;
    else 
        methylosmoleneText.style.opacity = 0.2
});


document.getElementById("months-linechart").addEventListener('change',function (){
    var month = document.getElementById("months-linechart").value;

    

    drawLineChart(month,sensor_no)
});