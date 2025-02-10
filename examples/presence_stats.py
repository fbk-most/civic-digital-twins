def tourist_presences_stats(weekday, season, weather):
    match season:
        case 'very high':
            mean = 5500
            std = 1500
        case 'high':
            mean = 4500
            std = 1500
        case 'mid':
            mean = 3000
            std = 1200
        case 'low':
            mean = 1500
            std = 1000
        case _:
            raise ValueError
    match weather:
        case 'bad':
            mean = mean / 2
            std = std / 2
        case 'unsettled' | 'good':
            pass
        case _:
            raise ValueError
    match weekday:
        case 'sunday':
            mean += 500
        case 'saturday':
            mean += 300
        case 'friday':
            mean += 100
        case 'monday' | 'tuesday' | 'wednesday' | 'thursday':
            pass
        case _:
            raise ValueError
    return {'mean': mean, 'std': std}

def excursionist_presences_stats(weekday, season, weather):
    return tourist_presences_stats(weekday, season, weather)