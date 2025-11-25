from django.http import JsonResponse, HttpResponseForbidden
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from location.models import Location
from .models import TripList, TripPath
from django.contrib import messages
from django.views.decorators.http import require_POST
from .TSP import Graph, distance
import json

# Create your views here.
@login_required
def favourite(request):
    if request.method == 'POST' and 'location_code' in request.POST:
        location_code = request.POST.get('location_code')

        if location_code:
            location = Location.objects.filter(code=location_code).first()
            if location and request.user in location.favourited_by.all():
                location.favourited_by.remove(request.user)
                messages.success(request, "Đã xoá địa điểm khỏi danh sách yêu thích.")

        return redirect('favourite')

    locations = Location.objects.filter(favourited_by=request.user)

    return render(request, "favourite.html", {
        'locations': locations
    })

@login_required
def my_trip(request):
    user = request.user
    trip_list_id = f"{user.username}-favourite"

    trip_list, _ = TripList.objects.get_or_create(id=trip_list_id, defaults={
        'user': user,
        'name': f"{user.username}'s Favourite Trip"
    })

    if request.method == 'POST':
        return handle_trip_creation(request, trip_list, user)

    return display_trip_list(request, trip_list)


def handle_trip_creation(request, trip_list, user):
    """Handle POST request to create a new trip path"""
    path_name = request.POST.get('path_name')
    if not path_name:
        return redirect('my_trip')

    selected_ids = request.POST.getlist('locations')
    if not selected_ids:
        messages.error(request, "Vui lòng chọn ít nhất một địa điểm.")
        return redirect('favourite')

    locations = list(Location.objects.filter(id__in=selected_ids, favourited_by=user))
    if not locations:
        messages.error(request, "Không tìm thấy các địa điểm đã chọn.")
        return redirect('favourite')

    location_mappings = create_location_mappings(locations)
    constraints = extract_constraints(request, locations, location_mappings['id_to_index'])
    graph_data = build_distance_graph(locations)

    path, cost = calculate_optimal_path(
        graph_data['graph'],
        constraints,
        location_mappings['id_to_index']
    )

    if path is None:
        messages.error(request, "Không thể tạo lịch trình hợp lệ với các ràng buộc đã chọn.")
        return redirect('favourite')

    create_trip_path(
        trip_list, path_name, path, cost,
        graph_data['durations_map'],
        location_mappings['index_to_id'],
        locations, constraints
    )

    unfavorite_locations(locations, user)
    return redirect('my_trip')


def create_location_mappings(locations):
    """Create bidirectional mappings between location IDs and indices"""
    id_to_index = {loc.id: idx for idx, loc in enumerate(locations)}
    index_to_id = {idx: loc.id for idx, loc in enumerate(locations)}
    return {'id_to_index': id_to_index, 'index_to_id': index_to_id}


def extract_constraints(request, locations, id_to_index):
    """Extract user constraints from POST data"""
    num_locations = len(locations)
    pinned_positions = [None] * num_locations
    fixed_position_flags = [False] * num_locations
    precedence_constraints = []

    start_id = parse_location_id(request.POST.get('start_point'))
    end_id = parse_location_id(request.POST.get('end_point'))

    for loc in locations:
        process_location_constraints(
            request, loc, id_to_index, num_locations,
            pinned_positions, fixed_position_flags, precedence_constraints
        )

    return {
        'pinned_positions': pinned_positions,
        'fixed_position_flags': fixed_position_flags,
        'precedence_constraints': precedence_constraints,
        'start_id': start_id,
        'end_id': end_id
    }


def parse_location_id(id_str):
    """Safely parse location ID from string"""
    return int(id_str) if id_str and id_str.isdigit() else None


def process_location_constraints(request, loc, id_to_index, num_locations,
                                 pinned_positions, fixed_position_flags, precedence_constraints):
    """Process pinned position and precedence constraints for a location"""
    loc_id_str = str(loc.id)
    index = id_to_index[loc.id]

    # Handle pinned position
    pinned_str = request.POST.get(f'pinned_order_{loc_id_str}')
    if pinned_str and pinned_str.isdigit():
        pinned_index = int(pinned_str) - 1
        if 0 <= pinned_index < num_locations:
            pinned_positions[pinned_index] = index
            fixed_position_flags[pinned_index] = True

    # Handle precedence constraint
    after_id_str = request.POST.get(f'precedence_after_{loc_id_str}')
    if after_id_str and after_id_str.isdigit():
        after_id = int(after_id_str)
        if after_id in id_to_index:
            precedence_constraints.append((id_to_index[after_id], index))


def build_distance_graph(locations):
    """Build graph with distances and durations between all locations"""
    coordinates = [loc.coordinate for loc in locations]
    distances = []
    durations_map = {}

    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            if i != j:
                dist, duration = distance(coordinates[i], coordinates[j])
                distances.append((i, j, dist))
                durations_map[(i, j)] = duration

    graph = Graph(len(locations))
    for u, v, w in distances:
        graph.add_edge(u, v, w)

    return {'graph': graph, 'durations_map': durations_map}


def calculate_optimal_path(graph, constraints, id_to_index):
    """Calculate optimal path using TSP algorithm"""
    start_index = id_to_index.get(constraints['start_id']) if constraints['start_id'] in id_to_index else None
    end_index = id_to_index.get(constraints['end_id']) if constraints['end_id'] in id_to_index else None

    return graph.find_hamiltonian_path(
        fixed_position=constraints['fixed_position_flags'],
        precedence_constraints=constraints['precedence_constraints'],
        start=start_index,
        end=end_index
    )


def create_trip_path(trip_list, path_name, path, cost, durations_map,
                     index_to_id, locations, constraints):
    """Create and save TripPath object"""
    total_duration = sum(
        durations_map.get((path[i], path[i+1]), 0) for i in range(len(path) - 1)
    )

    ordered_location_ids = [index_to_id[i] for i in path]

    start_point_obj = next(
        (loc for loc in locations if loc.id == constraints['start_id']), None
    )
    end_point_obj = next(
        (loc for loc in locations if loc.id == constraints['end_id']), None
    )

    TripPath.objects.create(
        trip_list=trip_list,
        path_name=path_name,
        locations_ordered=json.dumps(ordered_location_ids),
        total_distance=cost,
        total_duration=total_duration,
        start_point=start_point_obj,
        end_point=end_point_obj
    )


def unfavorite_locations(locations, user):
    """Remove locations from user's favorites"""
    for loc in locations:
        loc.favourited_by.remove(user)


def display_trip_list(request, trip_list):
    """Display user's trip paths"""
    trip_paths = TripPath.objects.filter(trip_list=trip_list).order_by('-created_at')
    parsed_trip_paths = parse_trip_paths(trip_paths)
    location_map = build_location_map(parsed_trip_paths)

    return render(request, 'my_trip.html', {
        'trip_paths': parsed_trip_paths,
        'location_map': location_map
    })


def parse_trip_paths(trip_paths):
    """Parse trip paths into display-ready format"""
    parsed_paths = []

    for path in trip_paths:
        try:
            loc_ids = json.loads(path.locations_ordered)
        except json.JSONDecodeError:
            loc_ids = []

        parsed_paths.append({
            'id': path.id,
            'path_name': path.path_name,
            'locations': loc_ids,
            'start_point': path.start_point.location if path.start_point else None,
            'end_point': path.end_point.location if path.end_point else None,
            'total_distance': round(path.total_distance / 1000, 1) if path.total_distance is not None else None,
            'total_duration': round(path.total_duration / 60, 1) if path.total_duration is not None else None,
            'created_at': path.created_at,
        })

    return parsed_paths


def build_location_map(parsed_trip_paths):
    """Build map of location IDs to names"""
    all_ids = []
    for path_data in parsed_trip_paths:
        all_ids.extend(path_data['locations'])

    location_qs = Location.objects.filter(id__in=all_ids)
    return {loc.id: loc.location for loc in location_qs}

@require_POST
@login_required
def delete_trip_path(request, path_id):
    if request.method != 'POST' or request.headers.get('x-requested-with') != 'XMLHttpRequest':
        return HttpResponseForbidden()
    trip_path = get_object_or_404(TripPath, pk=path_id)
    if trip_path.trip_list.user != request.user:
        return HttpResponseForbidden()
    trip_path.delete()
    return JsonResponse({'status': 'deleted'})