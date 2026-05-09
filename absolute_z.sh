#!/bin/bash

# Create a counter starting from -30
counter=-30

# Loop through trajectory files 0 to 61
for i in {0..61}; do
    # Create TCL script for each trajectory
    cat > analysis_${i}.tcl << EOF
#####Script developed by Kunal Rai#########
##for more detail read README####
#upload system psf 
mol new ../../input/lipid_aqp_wi.psf
#upload trajectory file
mol addfile ../windows_traj/traj_${i}.dcd waitfor all
# Define the atom selection here
set sel [atomselect top "residue 34649"]
# Open the output file for writing
set outfile [open "z_${counter}.dat" w]
# Write the header to the file
puts \$outfile "time z"
# Loop over all frames in the trajectory
set num_frames [molinfo top get numframes]
for {set frame 0} {\$frame < \$num_frames} {incr frame} {
    # Set the current frame
    animate goto \$frame
    
    # Update the atom selection
    \$sel frame \$frame
    
    # Calculate time
    set time [expr \$frame * 0.214]
    
    # Get the z-coordinate of the center of mass of the selected atoms
    #set z_coord [lindex [\$sel get {z}] 3]
    set z_coord [lindex [measure center \$sel weight mass] 2]
    
    # Write the time and z-coordinate to the file
    puts \$outfile "\$time \$z_coord"
}
# Close the output file
close \$outfile
# Delete the atom selection
\$sel delete
puts "Data has been written to z_${counter}.dat"
quit
EOF

    # Run VMD with the TCL script
    vmd -dispdev text -e analysis_${i}.tcl
    
    # Increment counter
    counter=$((counter + 1))
done
